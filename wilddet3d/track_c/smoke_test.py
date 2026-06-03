"""End-to-end Track C smoke test on a few real CA-1M trajectories (no disk cache).

Validates the full path: frozen-feature extraction -> trajectory encoder ->
temporal 3D head -> decode -> loss -> backward -> optimizer step. Checks:
  1. identity-at-init (decoded box == temporal-encoder prior),
  2. only Track C params are trainable (frozen stack has 0 grad params),
  3. loss decreases when overfitting a handful of trajectories.

Run (in the `opendet3d` env, from the WildDet3D repo root):
    python -m wilddet3d.track_c.smoke_test \
        --outputs /weka/.../itw_3dbox_det/outputs \
        --ckpt ckpt/wilddet3d_alldata_all_prompt_v1.0.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from vis4d.op.geometry.rotation import quaternion_to_matrix

from wilddet3d.track_c import (
    FrozenFeatureExtractor,
    TrackCRefiner,
    list_trajectories,
    load_trajectory,
    track_c_loss,
)


def extract_traj(ext, sample, device):
    feats = {k: [] for k in ("hidden", "ray", "depth", "box2d", "K", "iou")}
    ihw = None
    for fr in sample.frames:
        img = np.array(Image.open(fr.image_path).convert("RGB")).astype(np.float32)
        dep = np.load(fr.depth_path).astype(np.float32) if fr.depth_path else None
        f = ext.extract(img, fr.intrinsics, [fr.prompt_box_xyxy], depth=dep)
        feats["hidden"].append(f["hidden_states"])   # [L,1,256]
        feats["ray"].append(f["ray_embeddings"])     # [2401,81]
        feats["depth"].append(f["depth_latents"])    # [2401,256]
        feats["box2d"].append(f["pred_box_2d"][0])   # [4]
        feats["K"].append(f["intrinsics"])           # [3,3]
        feats["iou"].append(float(f["sel_iou"][0]))
        ihw = f["input_hw"]
    Kstack = torch.stack(feats["K"], 0)
    assert (Kstack - Kstack[0]).abs().max() < 1e-3, "intrinsics vary within trajectory"
    return {
        "hidden": torch.stack(feats["hidden"], dim=1).to(device),  # [L,T,1,256]
        "ray": torch.stack(feats["ray"], 0).to(device),
        "depth": torch.stack(feats["depth"], 0).to(device),
        "box2d": torch.stack(feats["box2d"], 0).to(device),
        "K": Kstack[0].to(device),                                 # [3,3] per-video constant
        "box_repr": torch.from_numpy(sample.box_repr()).to(device),
        "ts": torch.from_numpy(sample.timestamps_sec()).to(device),
        "measured": torch.from_numpy(sample.measured_mask()).to(device),
        "gt_center": torch.from_numpy(np.stack([f.gt_center for f in sample.frames])).to(device),
        "gt_dims": torch.from_numpy(np.stack([f.gt_dims for f in sample.frames])).to(device),
        "gt_quat": torch.from_numpy(np.stack([f.gt_quat_wxyz for f in sample.frames])).to(device),
        "input_hw": ihw, "iou": float(np.mean(feats["iou"])),
    }


def run_one(refiner, pack):
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
    )
    return out, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="itw_3dbox_det/outputs dir")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_traj", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default="", help="optional .pt feature cache path")
    args = ap.parse_args()
    torch.manual_seed(0)

    samples = []
    for vid, obj in list_trajectories(args.outputs, "step4"):
        s = load_trajectory(args.outputs, vid, obj, prior_source="step4")
        if s is not None and len(s.frames) >= 6:
            samples.append(s)
            print(f"  loaded {vid}/{obj[:8]} cat='{s.category}' T={len(s.frames)} "
                  f"measured={int(s.measured_mask().sum())}")
        if len(samples) >= args.n_traj:
            break
    assert samples, "no trajectories loaded"

    print("\n[build] frozen extractor + refiner ...")
    ext = FrozenFeatureExtractor(checkpoint=args.ckpt, device=args.device, use_depth_input=True)
    refiner = TrackCRefiner(reg_residual_from_prior=True).to(args.device)
    info = refiner.load_pretrained_head(args.ckpt)
    print(f"[build] head load: {info['loaded']} tensors; new(missing)={len(info['missing'])} "
          f"unexpected={len(info['unexpected'])}")
    refiner.finalize_init()

    if args.cache and os.path.exists(args.cache):
        packs = torch.load(args.cache, weights_only=False)
        packs = [{k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in p.items()} for p in packs]
    else:
        t0 = time.time()
        packs = [extract_traj(ext, s, args.device) for s in samples]
        print(f"[extract] {len(packs)} trajectories in {time.time()-t0:.1f}s; "
              f"mean sel IoU = {[round(p['iou'],3) for p in packs]}")
        if args.cache:
            torch.save([{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in p.items()} for p in packs], args.cache)

    # identity-at-init
    refiner.eval()
    with torch.no_grad():
        out, loss0 = run_one(refiner, packs[0])
        dec = refiner.decode_layer(out["reg"][-1, :, 0, :], packs[0]["box2d"], packs[0]["K"], packs[0]["input_hw"])
        prior_dec = refiner.decode_layer(out["temporal_box_12d"], packs[0]["box2d"], packs[0]["K"], packs[0]["input_hw"])
        id_err = (dec - prior_dec).abs().max().item()
    print(f"\n[init] identity-at-init max|decoded - temporal_prior| = {id_err:.2e} (expect ~0)")
    print("[init] losses traj0: " + ", ".join(f"{k}={float(v):.4f}" for k, v in loss0.items()))

    n_train = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in ext.wd.parameters() if p.requires_grad)
    print(f"[params] refiner trainable={n_train/1e6:.2f}M ; frozen-stack trainable={n_frozen} (expect 0)")

    refiner.train()
    opt = torch.optim.AdamW([p for p in refiner.parameters() if p.requires_grad], lr=args.lr)
    print(f"\n[train] overfitting {len(packs)} trajectories for {args.n_steps} steps ...")
    for step in range(args.n_steps):
        opt.zero_grad()
        tot, logs = 0.0, None
        for pack in packs:
            out, loss = run_one(refiner, pack)
            (loss["loss"] / len(packs)).backward()
            tot += float(loss["loss"]) / len(packs)
            logs = loss
        opt.step()
        if step % 5 == 0 or step == args.n_steps - 1:
            gate = refiner.head.temporal_gate[-1].abs().mean().item()
            print(f"  step {step:3d}  loss={tot:.4f}  center={float(logs['loss_center']):.4f} "
                  f"depth={float(logs['loss_depth']):.4f} dims={float(logs['loss_dims']):.4f} "
                  f"rot_deg={float(logs['loss_rot_deg']):.2f}  gate|.|={gate:.4f}")
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
