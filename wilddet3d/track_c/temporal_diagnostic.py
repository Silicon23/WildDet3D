"""Is the temporal channel actually doing anything? Two-part diagnostic.

(1) Weight diagnostics on a trained checkpoint:
    - temporal_gate magnitudes per (used) prediction layer  (a tiny gate caps the
      channel's contribution regardless of what the cross-attn learns)
    - TrajectoryEncoder + project_temporal + prompt_temporal Frobenius norms
    - per-layer EFFECTIVE perturbation `||gate * (updated_h - h)|| / ||h||`
      measured during a real forward (the actual fraction of the hidden state
      the temporal cross-attn block rewrites)

(2) Inference-time causal interventions on the temporal token:
    - `baseline`     : clean trajectory encoder output
    - `zero`         : temporal_tokens := 0   (model can still self-attend the layer)
    - `shuffle_frame`: tokens shuffled within each trajectory's time axis
    - `shuffle_traj` : each trajectory gets some OTHER trajectory's tokens
    - `random`       : tokens := N(0, std(real_tokens))
    - `mean`         : tokens := batch mean
    - `gates_zero`   : tokens unchanged, all `temporal_gate` weights set to 0
                      (removes the channel entirely; should reproduce no_temporal)

If `zero` / `shuffle_traj` / `random` match the baseline IoU to ~3 decimals, the
trained model has provably learned to ignore the temporal channel.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch
from vis4d.op.geometry.rotation import matrix_to_quaternion, quaternion_to_matrix

from wilddet3d.head.trajectory_encoder import TrajectoryEncoder
from wilddet3d.ops.iou_3d_safe import batch_box3d_iou
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import (
    CachedTrackCDataset,
    collate_trajs,
    list_cached_trajectories,
    split_by_video,
)
from wilddet3d.track_c.losses import geodesic_rotation_loss


def decode_per_traj(refiner, out, batch):
    """Decode final-layer boxes per trajectory (intrinsics constant within a video)."""
    reg = out["reg"][-1, :, 0, :].float()
    off, cs, ds, Rs = 0, [], [], []
    for n in batch["sizes"]:
        sl = slice(off, off + int(n)); off += int(n)
        K_traj = batch["K"][sl][0]
        dec = refiner.decode_layer(reg[sl], batch["box2d"][sl], K_traj, batch["input_hw"])
        cs.append(dec[:, 0:3]); ds.append(dec[:, 3:6])
        Rs.append(quaternion_to_matrix(dec[:, 6:10]))
    return torch.cat(cs, 0), torch.cat(ds, 0), torch.cat(Rs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--K", type=int, default=48, help="val trajectories sampled")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)

    refiner = TrackCRefiner(reg_residual_from_prior=False).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["refiner"]
    refiner.load_state_dict(sd, strict=True)
    refiner.eval()

    # -------- (1a) gate magnitudes --------
    print("\n=== Temporal gate (LayerScale, dim=256, applied as `gate * (updated - h)`) ===")
    print(f"{'layer':>5}  {'|g|.mean':>10}  {'|g|.max':>10}  {'std':>10}  note")
    for i in range(refiner.head.num_pred_layer):
        g = refiner.head.temporal_gate[i].detach()
        note = "USED" if i < 6 else "unused (7th clone)"
        print(f"{i:>5}  {g.abs().mean().item():>10.4f}  {g.abs().max().item():>10.4f}  "
              f"{g.std().item():>10.4f}  {note}")

    # -------- (1b) module weight magnitudes --------
    def total_norm(mod):
        return sum(p.pow(2).sum().item() for p in mod.parameters()) ** 0.5
    print(f"\ntraj_encoder ||W||_F (total):                     {total_norm(refiner.traj_encoder):.3f}")
    print(f"  box_decode[-1] (zero-init final Linear) ||W||:   {refiner.traj_encoder.box_decode[-1].weight.norm().item():.4f}")
    print(f"  box_decode[-1] bias ||b||:                       {refiner.traj_encoder.box_decode[-1].bias.norm().item():.4f}")
    print(f"project_temporal[0] ||W||_F:                       {total_norm(refiner.head.project_temporal[0]):.3f}")
    print(f"prompt_temporal[0] ||W||_F:                        {total_norm(refiner.head.prompt_temporal[0]):.3f}")

    # -------- (2) load K val trajectories from the same by-video val split --------
    paths = list_cached_trajectories(args.cache_dir)
    _, val_paths, _ = split_by_video(paths, args.val_frac)
    sub = val_paths[: args.K]
    ds = CachedTrackCDataset(args.cache_dir, sub, preload=True)
    packs = [ds[i] for i in range(len(ds))]
    batch = collate_trajs(packs)
    batch_dev = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    print(f"\n=== Loaded {len(sub)} val trajectories "
          f"({sum(batch['sizes'])} total frames) ===")

    # -------- (1c) per-layer effective perturbation (measured during baseline) --------
    perturb = {}
    def make_hook(i):
        def hook(module, inputs, output):
            h = inputs[0]                                 # [B, S, 256]
            u = output                                    # [B, S, 256]
            gate = refiner.head.temporal_gate[i]          # [256]
            contrib = (gate * (u - h)).norm(dim=-1)       # [B, S]
            hn = h.norm(dim=-1).clamp(min=1e-6)
            perturb[i] = (contrib / hn).mean().item()
        return hook
    handles = [refiner.head.prompt_temporal[i].register_forward_hook(make_hook(i))
               for i in range(6)]

    # -------- (2) ablations --------
    orig_te_forward = TrajectoryEncoder.forward

    def ablation(mode, seed=42):
        def forward(self, box_repr, ts, measured_mask, **kw):
            tokens, box_out = orig_te_forward(self, box_repr, ts, measured_mask, **kw)
            g = torch.Generator(device=tokens.device).manual_seed(seed)
            if mode == "baseline":
                return tokens, box_out
            if mode == "zero":
                return torch.zeros_like(tokens), box_out
            if mode == "shuffle_frame":
                t = tokens.clone()
                for b in range(t.shape[0]):
                    p = torch.randperm(t.shape[1], device=t.device, generator=g)
                    t[b] = t[b, p]
                return t, box_out
            if mode == "shuffle_traj":
                p = torch.randperm(tokens.shape[0], device=tokens.device, generator=g)
                return tokens[p], box_out
            if mode == "random":
                std = tokens.float().std().item()
                return (torch.randn(tokens.shape, device=tokens.device,
                                    generator=g, dtype=torch.float32) * std).to(tokens.dtype), box_out
            if mode == "mean":
                mn = tokens.mean(dim=(0, 1), keepdim=True)
                return mn.expand_as(tokens), box_out
            raise ValueError(mode)
        return forward

    def run_once(mode):
        TrajectoryEncoder.forward = ablation(mode)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = refiner.forward_vectorized(batch_dev)
        pc, pd, pR = decode_per_traj(refiner, out, batch_dev)
        valid = batch_dev["gt_center"][:, 2] > 1e-3
        gt_R = quaternion_to_matrix(batch_dev["gt_quat"])
        p_quat = matrix_to_quaternion(pR)
        p10 = torch.cat([pc, pd, p_quat], -1)[valid]
        g10 = torch.cat([batch_dev["gt_center"], batch_dev["gt_dims"],
                         batch_dev["gt_quat"]], -1)[valid]
        iou = batch_box3d_iou(p10, g10).mean().item()
        c = (pc - batch_dev["gt_center"])[valid].norm(dim=-1).mean().item()
        r = geodesic_rotation_loss(pR[valid], gt_R[valid], True).mean().item() * 180.0 / math.pi
        return iou, c, r

    print(f"\n=== Inference-time intervention table  ({len(sub)} trajs) ===")
    print(f"{'ablation':>16}  {'iou3d':>7}  {'center_m':>10}  {'rot_deg':>9}  ΔIoU")
    base_iou, base_c, base_r = run_once("baseline")
    print(f"{'baseline':>16}  {base_iou:>7.4f}  {base_c:>10.4f}  {base_r:>9.3f}  {'  --':>6}")
    for mode in ["zero", "shuffle_frame", "shuffle_traj", "random", "mean"]:
        iou, c, r = run_once(mode)
        print(f"{mode:>16}  {iou:>7.4f}  {c:>10.4f}  {r:>9.3f}  {iou - base_iou:+.4f}")

    # gates_zero: leave tokens alone, force every gate to zero
    TrajectoryEncoder.forward = orig_te_forward
    saved = [g.data.clone() for g in refiner.head.temporal_gate]
    for g in refiner.head.temporal_gate:
        g.data.zero_()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = refiner.forward_vectorized(batch_dev)
    pc, pd, pR = decode_per_traj(refiner, out, batch_dev)
    valid = batch_dev["gt_center"][:, 2] > 1e-3
    gt_R = quaternion_to_matrix(batch_dev["gt_quat"])
    p_quat = matrix_to_quaternion(pR)
    p10 = torch.cat([pc, pd, p_quat], -1)[valid]
    g10 = torch.cat([batch_dev["gt_center"], batch_dev["gt_dims"],
                     batch_dev["gt_quat"]], -1)[valid]
    iou = batch_box3d_iou(p10, g10).mean().item()
    c = (pc - batch_dev["gt_center"])[valid].norm(dim=-1).mean().item()
    r = geodesic_rotation_loss(pR[valid], gt_R[valid], True).mean().item() * 180.0 / math.pi
    print(f"{'gates_zero':>16}  {iou:>7.4f}  {c:>10.4f}  {r:>9.3f}  {iou - base_iou:+.4f}")
    for g, s in zip(refiner.head.temporal_gate, saved):
        g.data.copy_(s)
    for h in handles:
        h.remove()

    print("\n=== Per-layer effective perturbation `||gate * (updated_h - h)|| / ||h||` ===")
    print("(actual fraction of the hidden state the temporal block rewrites at each layer)")
    for i in range(6):
        print(f"  layer {i}: {perturb[i]:.5f}")

    print("\nInterpretation: if `zero` / `shuffle_traj` / `random` all match `baseline`")
    print("on iou3d within ~0.001, the model provably ignores the temporal channel.")
    print("Per-layer perturbation < 0.01 means the channel is at most cosmetically active.")


if __name__ == "__main__":
    main()
