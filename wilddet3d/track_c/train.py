"""Track C training + eval on the precomputed feature cache.

Split by VIDEO. Trains only the TrackCRefiner (trajectory encoder + temporal 3D
head); the frozen stack lives entirely in the cache. Eval reports per-frame
center / dims / rotation error for three box sources so we can see whether the
image-grounded refiner beats its own prior:
  - ``input``   : the Step-4 (FoundationPose) prior box (what we anchor to)
  - ``traj_enc``: the trajectory-encoder densified/smoothed box (no image)
  - ``track_c`` : the full image-grounded refiner output (final head layer)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from vis4d.op.geometry.rotation import matrix_to_quaternion, quaternion_to_matrix

from wilddet3d.ops.iou_3d_safe import batch_box3d_iou
from wilddet3d.ops.rotation import rotation_6d_to_matrix
from wilddet3d.track_c import TrackCRefiner, track_c_loss
from wilddet3d.track_c.dataset import (
    CachedTrackCDataset,
    VideoGroupedSampler,
    list_cached_trajectories,
    split_by_video,
)
from wilddet3d.track_c.losses import geodesic_rotation_loss


def to_dev(pack, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in pack.items()}


def forward_loss(refiner, pack, loss_kw):
    out = refiner(
        hidden_states=pack["hidden"], ray_embeddings=pack["ray"],
        depth_latents=pack["depth"], pred_box_2d=pack["box2d"],
        intrinsics=pack["K"], box_repr=pack["box_repr"],
        timestamps=pack["ts"], measured_mask=pack["measured"],
        input_hw=pack["input_hw"],
    )
    gt_R = quaternion_to_matrix(pack["gt_quat"])
    loss = track_c_loss(
        out["reg"], refiner.coder, pack["gt_center"], pack["gt_dims"],
        pack["gt_quat"], gt_R, pack["box2d"], pack["K"], pack["input_hw"],
        valid=(pack["gt_center"][:, 2] > 1e-3), **loss_kw,
    )
    return out, loss


def _box_repr_to_cdr(box_repr):
    """[T,12]=[center,log_dims,rot6d] -> center[T,3], dims[T,3], R[T,3,3]."""
    center = box_repr[:, 0:3]
    dims = torch.exp(box_repr[:, 3:6])
    R = rotation_6d_to_matrix(box_repr[:, 6:12])
    return center, dims, R


@torch.no_grad()
def evaluate(refiner, loader, dev):
    refiner.eval()
    agg = {s: {"center": 0.0, "dims": 0.0, "rot": 0.0, "iou": 0.0}
           for s in ("input", "traj_enc", "track_c")}
    nfr = 0
    for pack in loader:
        pack = to_dev(pack, dev)
        out, _ = forward_loss(refiner, pack, {})
        gt_c, gt_d = pack["gt_center"], pack["gt_dims"]
        gt_R = quaternion_to_matrix(pack["gt_quat"])
        valid = (gt_c[:, 2] > 1e-3)
        n = int(valid.sum())
        if n == 0:
            continue
        nfr += n
        gt_box10 = torch.cat([gt_c, gt_d, pack["gt_quat"]], dim=-1)  # [T,10]

        # input prior
        c, d, R = _box_repr_to_cdr(pack["box_repr"])
        srcs = {"input": (c, d, R)}
        # trajectory-encoder box
        c2, d2, R2 = _box_repr_to_cdr(out["traj_box_out"])
        srcs["traj_enc"] = (c2, d2, R2)
        # track C decoded final layer
        dec = refiner.decode_layer(out["reg"][-1, :, 0, :], pack["box2d"], pack["K"], pack["input_hw"])
        cc, dd = dec[:, 0:3], dec[:, 3:6]
        RR = quaternion_to_matrix(dec[:, 6:10])
        srcs["track_c"] = (cc, dd, RR)

        for s, (pc, pd, pR) in srcs.items():
            ce = (pc - gt_c).norm(dim=-1)[valid].sum().item()
            de = (pd - gt_d).abs().mean(dim=-1)[valid].sum().item()
            re = geodesic_rotation_loss(pR, gt_R, True)[valid].sum().item()
            pquat = matrix_to_quaternion(pR)
            pbox10 = torch.cat([pc, pd, pquat], dim=-1)
            iou = batch_box3d_iou(pbox10[valid], gt_box10[valid]).sum().item()
            agg[s]["center"] += ce
            agg[s]["dims"] += de
            agg[s]["rot"] += re
            agg[s]["iou"] += iou
    out = {}
    for s in agg:
        out[s] = {
            "center_m": agg[s]["center"] / max(nfr, 1),
            "dims_m": agg[s]["dims"] / max(nfr, 1),
            "rot_deg": agg[s]["rot"] / max(nfr, 1) * 180.0 / math.pi,
            "iou3d": agg[s]["iou"] / max(nfr, 1),
        }
    out["_n_frames"] = nfr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="WildDet3D ckpt for pretrained 3D head")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8, help="trajectories per optimizer step")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--gate_lr_mult", type=float, default=20.0,
                    help="LR multiplier for the new temporal modules (gate/prompt)")
    ap.add_argument("--reg_residual_from_prior", type=int, default=1)
    ap.add_argument("--w_center", type=float, default=1.0)
    ap.add_argument("--w_depth", type=float, default=1.0)
    ap.add_argument("--w_dims", type=float, default=1.0)
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device

    paths = list_cached_trajectories(args.cache_dir)
    if args.limit:
        paths = paths[:args.limit]
    train_paths, val_paths, val_vids = split_by_video(paths, args.val_frac)
    print(f"[data] {len(paths)} trajs -> train {len(train_paths)} / val {len(val_paths)} "
          f"({len(val_vids)} val videos)", flush=True)

    train_ds = CachedTrackCDataset(args.cache_dir, train_paths)
    val_ds = CachedTrackCDataset(args.cache_dir, val_paths)
    train_sampler = VideoGroupedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=None, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=4)

    refiner = TrackCRefiner(reg_residual_from_prior=bool(args.reg_residual_from_prior)).to(dev)
    info = refiner.load_pretrained_head(args.ckpt)
    print(f"[build] head load: {info['loaded']} tensors, new={len(info['missing'])}", flush=True)
    refiner.finalize_init()

    # Higher LR for the new temporal modules (esp. the zero-init LayerScale gate)
    # so the image-grounded temporal-prompt pathway activates from a cold start,
    # while preserving identity-at-init (gate starts at exactly 0).
    temporal_params, base_params = [], []
    for name, p in refiner.named_parameters():
        if not p.requires_grad:
            continue
        if ("temporal_gate" in name or "prompt_temporal" in name
                or "project_temporal" in name):
            temporal_params.append(p)
        else:
            base_params.append(p)
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": temporal_params, "lr": args.lr * args.gate_lr_mult},
    ], weight_decay=1e-4)
    print(f"[build] param groups: base={sum(p.numel() for p in base_params)/1e6:.2f}M "
          f"temporal={sum(p.numel() for p in temporal_params)/1e6:.2f}M "
          f"(temporal lr x{args.gate_lr_mult})", flush=True)
    total_steps = args.epochs * max(1, len(train_paths) // args.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    loss_kw = dict(w_center=args.w_center, w_depth=args.w_depth,
                   w_dims=args.w_dims, w_rot=args.w_rot)

    print("[eval] baseline (init / before training):", flush=True)
    ev = evaluate(refiner, val_loader, dev)
    print("  " + json.dumps(ev), flush=True)

    best = 1e9
    for ep in range(args.epochs):
        refiner.train()
        train_sampler.set_epoch(ep)
        t0 = time.time()
        running, ncount, accum_n = 0.0, 0, 0
        opt.zero_grad()
        logs = None
        for pack in train_loader:
            pack = to_dev(pack, dev)
            out, loss = forward_loss(refiner, pack, loss_kw)
            (loss["loss"] / args.accum).backward()
            running += float(loss["loss"]); ncount += 1; accum_n += 1
            logs = loss
            if accum_n >= args.accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in refiner.parameters() if p.requires_grad], 5.0)
                opt.step(); sched.step(); opt.zero_grad(); accum_n = 0
        if accum_n > 0:
            opt.step(); opt.zero_grad()
        dt = time.time() - t0
        gate = refiner.head.temporal_gate[-1].abs().mean().item()
        print(f"[ep {ep}] train_loss={running/max(ncount,1):.4f} "
              f"(center={float(logs['loss_center']):.3f} depth={float(logs['loss_depth']):.3f} "
              f"dims={float(logs['loss_dims']):.3f} rot_deg={float(logs['loss_rot_deg']):.2f}) "
              f"gate|.|={gate:.4f} lr={sched.get_last_lr()[0]:.2e} {dt:.0f}s", flush=True)

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            ev = evaluate(refiner, val_loader, dev)
            tc = ev["track_c"]; inp = ev["input"]
            print(f"[ep {ep}] EVAL track_c: iou3d={tc['iou3d']:.3f} center={tc['center_m']:.3f}m "
                  f"dims={tc['dims_m']:.3f}m rot={tc['rot_deg']:.2f}deg | input(step4): "
                  f"iou3d={inp['iou3d']:.3f} center={inp['center_m']:.3f}m dims={inp['dims_m']:.3f}m "
                  f"rot={inp['rot_deg']:.2f}deg | nfr={ev['_n_frames']}",
                  flush=True)
            score = -tc["iou3d"]  # maximize 3D IoU (lower score = better)
            if score < best:
                best = score
                torch.save({"refiner": refiner.state_dict(), "args": vars(args),
                            "epoch": ep, "eval": ev}, f"{args.out_dir}/best.pt")
                print(f"[ep {ep}] saved best (iou3d={tc['iou3d']:.4f})", flush=True)
    torch.save({"refiner": refiner.state_dict(), "args": vars(args)},
               f"{args.out_dir}/last.pt")
    print("[train] done", flush=True)


if __name__ == "__main__":
    main()
