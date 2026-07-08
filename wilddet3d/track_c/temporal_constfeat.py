"""Check whether prompt_temporal[i] has become a query-independent constant feature.

For each USED prediction layer i (0..5):
  - take the actual (hidden_state h, temporal_tokens T, ray, depth) seen by the
    layer during a real forward on 12 val trajectories (concat sumT frames),
  - measure the layer's OUTPUT `u = prompt_temporal[i](h, T, T)` — what the
    cross-attn block emits BEFORE the LayerScale gate,
  - report `u`'s coefficient of variation across queries
    (std over queries / mean magnitude). Low CV  =>  output is ~constant
    across queries  =>  cross-attn collapsed to a layer-wise constant feature
    that doesn't depend on the temporal token's content.

Also runs the same `gates_zero` ablation on v3 + v10 (both trained with
`--no_temporal 1` so gates were frozen at 0 throughout). Predicted no-op there
— if confirmed, v8's gates_zero degradation is uniquely from a *trained*
constant feature, not a generic temporal-module effect.
"""
from __future__ import annotations

import argparse
import math

import torch
from vis4d.op.geometry.rotation import matrix_to_quaternion, quaternion_to_matrix

from wilddet3d.ops.iou_3d_safe import batch_box3d_iou
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import (
    CachedTrackCDataset,
    collate_trajs,
    list_cached_trajectories,
    split_by_video,
)
from wilddet3d.track_c.losses import geodesic_rotation_loss


def load_refiner(ckpt, device):
    r = TrackCRefiner(
        reg_residual_from_prior=False,
        use_temporal_kv_norm=True,
        temporal_multi_token=True,
    ).to(device)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["refiner"]
    r.load_state_dict(sd, strict=True)
    r.eval()
    return r


def decode_per_traj(refiner, out, batch):
    reg = out["reg"][-1, :, 0, :].float()
    off, cs, ds, Rs = 0, [], [], []
    for n in batch["sizes"]:
        sl = slice(off, off + int(n)); off += int(n)
        K = batch["K"][sl][0]
        d = refiner.decode_layer(reg[sl], batch["box2d"][sl], K, batch["input_hw"])
        cs.append(d[:, 0:3]); ds.append(d[:, 3:6])
        Rs.append(quaternion_to_matrix(d[:, 6:10]))
    return torch.cat(cs, 0), torch.cat(ds, 0), torch.cat(Rs, 0)


def metrics(refiner, batch):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = refiner.forward_vectorized(batch)
    pc, pd, pR = decode_per_traj(refiner, out, batch)
    v = batch["gt_center"][:, 2] > 1e-3
    gtR = quaternion_to_matrix(batch["gt_quat"])
    p10 = torch.cat([pc, pd, matrix_to_quaternion(pR)], -1)[v]
    g10 = torch.cat([batch["gt_center"], batch["gt_dims"], batch["gt_quat"]], -1)[v]
    iou = batch_box3d_iou(p10, g10).mean().item()
    c = (pc - batch["gt_center"])[v].norm(dim=-1).mean().item()
    r = geodesic_rotation_loss(pR[v], gtR[v], True).mean().item() * 180.0 / math.pi
    return iou, c, r


def gates_zero_test(refiner, batch, label):
    base = metrics(refiner, batch)
    saved = [g.data.clone() for g in refiner.head.temporal_gate]
    for g in refiner.head.temporal_gate:
        g.data.zero_()
    zer = metrics(refiner, batch)
    for g, s in zip(refiner.head.temporal_gate, saved):
        g.data.copy_(s)
    print(f"{label:>14}  base  iou3d={base[0]:.4f}  c={base[1]:.4f}  r={base[2]:.3f}")
    print(f"{label:>14}  gates_zero  iou3d={zer[0]:.4f}  c={zer[1]:.4f}  r={zer[2]:.3f}"
          f"  ΔIoU={zer[0]-base[0]:+.4f}")


def constfeat_check(refiner, batch, layer_ids=range(6)):
    """For each layer, capture (h_in, u_out) at prompt_temporal[i].
       Cosine-and-RMS check: is u_out ~constant across the sumT queries?"""
    caps = {}
    def make_hook(i):
        def hook(mod, inputs, output):
            h = inputs[0].float()                       # [B, S, 256]
            u = output.float()                          # [B, S, 256]
            x = u.reshape(-1, u.shape[-1])              # [N, 256]
            mu = x.mean(dim=0)                          # [256]
            sd = x.std(dim=0)                           # [256]
            ratio = (sd.norm() / (mu.norm() + 1e-12)).item()
            xn = torch.nn.functional.normalize(x, dim=-1)
            cos = (xn @ xn.t()).triu(1)
            n = xn.shape[0]
            cos_mean = (cos.sum() * 2 / (n * (n - 1))).item()
            caps[i] = dict(mu_norm=mu.norm().item(), std_norm=sd.norm().item(),
                           ratio=ratio, cos_mean=cos_mean, N=n)
        return hook
    handles = [refiner.head.prompt_temporal[i].register_forward_hook(make_hook(i))
               for i in layer_ids]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        refiner.forward_vectorized(batch)
    for h in handles:
        h.remove()
    print(f"{'layer':>5}  {'||mu||':>10}  {'||std||':>10}  {'std/mu':>10}  "
          f"{'mean_cos':>10}  N={caps[0]['N']}")
    for i in layer_ids:
        c = caps[i]
        print(f"{i:>5}  {c['mu_norm']:>10.3f}  {c['std_norm']:>10.3f}  "
              f"{c['ratio']:>10.4f}  {c['cos_mean']:>10.4f}")
    print("std/mu small AND mean_cos near 1.0  =>  near-constant output across queries.")


def load_val_batch(cache_dir, val_frac, K, dev):
    paths = list_cached_trajectories(cache_dir)
    _, val_paths, _ = split_by_video(paths, val_frac)
    ds = CachedTrackCDataset(cache_dir, val_paths[:K], preload=True)
    packs = [ds[i] for i in range(len(ds))]
    batch = collate_trajs(packs)
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--v10", default="")
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)

    batch = load_val_batch(args.cache_dir, args.val_frac, args.K, args.device)
    print(f"\n=== Loaded {args.K} val trajectories, {sum(batch['sizes'])} frames ===")

    # ---- (A) constant-feature check on v8 ----
    print("\n=== (A) v8: is prompt_temporal[i](h, T, T) ~constant across queries? ===")
    r8 = load_refiner(args.v8, args.device)
    constfeat_check(r8, batch)

    # ---- (B) gates_zero cross-check on v3 + v10 (both had gates frozen at 0) ----
    print("\n=== (B) gates_zero cross-check: should be a no-op on v3/v10 ===")
    print("(if gates were frozen at 0 during training, zeroing them now changes nothing)")
    gates_zero_test(r8, batch, "v8")
    r3 = load_refiner(args.v3, args.device)
    gates_zero_test(r3, batch, "v3")
    if args.v10:
        try:
            r10 = load_refiner(args.v10, args.device)
            gates_zero_test(r10, batch, "v10")
        except FileNotFoundError:
            print("(v10 best.pt not yet present — still training)")


if __name__ == "__main__":
    main()
