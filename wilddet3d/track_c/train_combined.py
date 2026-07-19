"""Combined-dataset Track C training: CA-1M + Waymo (veh+cyc, ped) + ADT.

Consumes a JSON config listing per-source {name, cache_dir, val_split_file,
pairing_index, categories, weight, val_frac}. Uses CombinedTrackCDataset +
WeightedMultiSourceSampler for within-batch weighted mixing (35/25/40 CA-1M/
Waymo/ADT anchor per the 2026-07-02 design). Runs 3 (or N) per-dataset val
loaders + a weighted-unified iou3d for model selection.

CLI mirrors train.py's flags for the model / loss config; the multi-source
knobs are:
    --datasets_json PATH   config file (schema below)
    --resume                              (via --auto_resume, standard)

Datasets JSON schema:
{
  "sources": [
    {"name": "ca1m", "cache_dir": ".../track_c_feature_cache",
     "val_split_file": ".../val_split.json", "val_frac": 0.05,
     "pairing_index": null, "categories": "", "weight": 0.35},
    ...
  ]
}
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

from wilddet3d.ops.rotation import rotation_6d_to_matrix
from wilddet3d.ops.iou_3d_safe import batch_box3d_iou
from wilddet3d.track_c import TrackCRefiner, track_c_loss
from wilddet3d.track_c.dataset import (
    CachedTrackCDataset,
    CombinedTrackCDataset,
    WeightedMultiSourceSampler,
    build_per_source_val_datasets,
    collate_trajs,
)
from wilddet3d.track_c.losses import (
    build_targets,
    derivative_matching_loss,
    encode_targets_batched,
    geodesic_rotation_loss,
    track_c_loss_from_targets,
)
from wilddet3d.track_c.smoothness_losses import pattern_smoothness_loss


# ---------- reusable pieces (small; copied to avoid coupling to train.py) ----

def to_dev(pack, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in pack.items()}


def _box_repr_to_cdr(box_repr):
    center = box_repr[:, 0:3]
    dims = torch.exp(box_repr[:, 3:6])
    R = rotation_6d_to_matrix(box_repr[:, 6:12])
    return center, dims, R


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


@torch.no_grad()
def evaluate_single(refiner, loader, dev):
    """Per-source val: same shape as train.py's evaluate()."""
    refiner.eval()
    agg = {s: {"center": 0.0, "dims": 0.0, "rot": 0.0, "iou": 0.0}
           for s in ("input", "track_c")}
    nfr = 0
    for pack in loader:
        pack = to_dev(pack, dev)
        # Cache stores hidden/ray/depth in bf16 (CPU-transfer speed); training
        # autocasts around the forward. Eval doesn't autocast, and torch >=2.11
        # (B300 upgrade 2026-07-08) no longer silently promotes bf16 activations
        # for fp32 LayerNorm params → RuntimeError. Cast to fp32 here; works on
        # both old and new torch, keeps eval math in fp32 (no accuracy risk).
        for k in ("hidden", "ray", "depth"):
            pack[k] = pack[k].float()
        out, _ = forward_loss(refiner, pack, {})
        gt_c, gt_d = pack["gt_center"], pack["gt_dims"]
        gt_R = quaternion_to_matrix(pack["gt_quat"])
        valid = (gt_c[:, 2] > 1e-3)
        n = int(valid.sum())
        if n == 0:
            continue
        nfr += n
        gt_box10 = torch.cat([gt_c, gt_d, pack["gt_quat"]], dim=-1)

        c, d, R = _box_repr_to_cdr(pack["box_repr"])
        srcs = {"input": (c, d, R)}
        dec = refiner.decode_layer(out["reg"][-1, :, 0, :], pack["box2d"],
                                    pack["K"], pack["input_hw"])
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


def evaluate_multi(refiner, val_loaders_by_name, dev, weights_by_name):
    """Run evaluate_single per source, plus a weighted-unified iou3d."""
    per = {name: evaluate_single(refiner, loader, dev)
           for name, loader in val_loaders_by_name.items()}
    # weighted unified: user weights (design intent) not n_frames (raw prevalence)
    wsum = sum(weights_by_name.values())
    unified_iou = sum(weights_by_name[n] * per[n]["track_c"]["iou3d"]
                       for n in per) / max(wsum, 1e-9)
    unified_ce = sum(weights_by_name[n] * per[n]["track_c"]["center_m"]
                       for n in per) / max(wsum, 1e-9)
    unified_rot = sum(weights_by_name[n] * per[n]["track_c"]["rot_deg"]
                       for n in per) / max(wsum, 1e-9)
    return {
        "per_source": per,
        "unified": {"iou3d": unified_iou, "center_m": unified_ce,
                     "rot_deg": unified_rot},
    }


# ---------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_json", required=True,
                    help="path to JSON config with 'sources' list "
                         "({name, cache_dir, val_split_file, pairing_index, "
                         "categories, weight, val_frac})")
    ap.add_argument("--ckpt", required=True, help="WildDet3D pretrained head")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_from", default="",
                    help="checkpoint (best.pt or last.pt) to warm-start full "
                         "refiner from, before training. Use WX3_adt/best.pt.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_trajs", type=int, default=8)
    ap.add_argument("--max_frames_per_batch", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_lr_mult", type=float, default=20.0)
    ap.add_argument("--reg_residual_from_prior", type=int, default=0)
    ap.add_argument("--no_traj_encoder", type=int, default=0)
    ap.add_argument("--use_layer_bias", type=int, default=0)
    ap.add_argument("--temporal_kv_norm", type=int, default=1)
    ap.add_argument("--warm_start_temporal", type=int, default=0)
    ap.add_argument("--temporal_multi_token", type=int, default=1)
    ap.add_argument("--temporal_block", default="xattn_only",
                    choices=["prompt3d", "xattn_only"])
    ap.add_argument("--mask_frame_p", type=float, default=0.25)
    ap.add_argument("--w_center", type=float, default=1.0)
    ap.add_argument("--w_depth", type=float, default=1.0)
    ap.add_argument("--w_dims", type=float, default=1.0)
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--w_vel", type=float, default=0.0)
    ap.add_argument("--w_acc", type=float, default=0.0)
    ap.add_argument("--w_rotvel", type=float, default=0.0)
    ap.add_argument("--w_rotvel2", type=float, default=0.0)
    ap.add_argument("--rotvel2_form", default="chordal", choices=["chordal", "geodesic"])
    ap.add_argument("--w_acc_hinge", type=float, default=0.0)
    ap.add_argument("--acc_hinge_margin", type=float, default=0.0)
    ap.add_argument("--w_pos_flip", type=float, default=0.0)
    ap.add_argument("--w_rot_flip", type=float, default=0.0)
    ap.add_argument("--flip_lags", default="1,2")
    ap.add_argument("--pos_delta_mm", type=float, default=10.0)
    ap.add_argument("--rot_delta_deg", type=float, default=0.05)
    ap.add_argument("--flip_margin", type=float, default=0.2)
    ap.add_argument("--w_l7", type=float, default=0.0)
    ap.add_argument("--w_l8", type=float, default=0.0)
    ap.add_argument("--w_l9a", type=float, default=0.0)
    ap.add_argument("--w_l9b", type=float, default=0.0)
    ap.add_argument("--spec_window", type=int, default=16)
    ap.add_argument("--spec_hop", type=int, default=8)
    ap.add_argument("--spec_cutoff_period", type=float, default=4.0)
    ap.add_argument("--spec_delta_mm", type=float, default=50.0)
    ap.add_argument("--spec_l9b_margin", type=float, default=0.05)
    ap.add_argument("--warmup_steps", type=int, default=200,
                    help="linear LR warmup over this many optimizer steps "
                         "(then cosine). Set 0 to disable. Guards against "
                         "the initial-domain-shock NaN when the pretrained "
                         "head produces catastrophic predictions on new caches.")
    ap.add_argument("--skip_bad_steps", type=int, default=1,
                    help="skip optimizer step (and grad reset) if loss is "
                         "NaN or inf. Prevents one bad batch from corrupting "
                         "all subsequent weights.")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--auto_resume", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.seed)

    # ---------- load config, build combined dataset + val loaders ---------
    with open(args.datasets_json) as f:
        cfg = json.load(f)
    sources = cfg["sources"]
    weights = [s["weight"] for s in sources]
    names = [s["name"] for s in sources]
    weights_by_name = dict(zip(names, weights))
    print(f"[cfg] {len(sources)} sources, weights: "
          f"{', '.join(f'{n}={w:.3f}' for n,w in zip(names, weights))}",
          flush=True)

    print("[data] building combined dataset + preloading each source ...", flush=True)
    train_ds = CombinedTrackCDataset(sources, preload=True)
    val_dss = build_per_source_val_datasets(train_ds, preload=True)
    print(f"[data] combined train: {len(train_ds)} trajs across {len(sources)} sources",
          flush=True)
    for n, vds in val_dss.items():
        print(f"[data] val[{n}]: {len(vds)} trajs", flush=True)

    train_sampler = WeightedMultiSourceSampler(
        train_ds, weights, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_ds, sampler=train_sampler,
                               batch_size=None, num_workers=0)
    val_loaders = {n: DataLoader(vds, batch_size=None, num_workers=0)
                    for n, vds in val_dss.items()}

    # ---------- model / opt (mirrors train.py) ----------------------------
    refiner = TrackCRefiner(
        reg_residual_from_prior=bool(args.reg_residual_from_prior),
        use_temporal_modules=not bool(args.no_traj_encoder),
        use_layer_bias=bool(args.use_layer_bias),
        use_temporal_kv_norm=bool(args.temporal_kv_norm),
        temporal_multi_token=bool(args.temporal_multi_token),
        temporal_block=args.temporal_block,
    ).to(dev)
    info = refiner.load_pretrained_head(args.ckpt)
    print(f"[build] head load: {info['loaded']} tensors, new={len(info['missing'])}",
          flush=True)
    refiner.finalize_init()
    if args.warm_start_temporal:
        refiner.warm_start_temporal_from_depth()
        print("[build] warm-started temporal branch from pretrained depth branch",
              flush=True)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu",
                        weights_only=False)["refiner"]
        missing, unexpected = refiner.load_state_dict(sd, strict=False)
        print(f"[build] warm-started full refiner from {args.init_from} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    flip_lags = tuple(int(x) for x in str(args.flip_lags).split(",") if x.strip())

    temporal_params, base_params = [], []
    for name, p in refiner.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("temporal_gate", "prompt_temporal",
                                     "project_temporal", "temporal_kv_norm",
                                     "layer_bias")):
            temporal_params.append(p)
        else:
            base_params.append(p)
    pg = [{"params": base_params, "lr": args.lr}]
    if temporal_params:
        pg.append({"params": temporal_params, "lr": args.lr * args.gate_lr_mult})
    opt = torch.optim.AdamW(pg, weight_decay=1e-4)
    print(f"[build] param groups: base={sum(p.numel() for p in base_params)/1e6:.2f}M "
          f"temporal={sum(p.numel() for p in temporal_params)/1e6:.2f}M "
          f"(temporal lr x{args.gate_lr_mult})", flush=True)
    total_steps = args.epochs * max(1, len(train_ds) // args.batch_trajs)
    # Warmup guards against initial-domain-shock NaN when the pretrained head
    # produces catastrophic predictions on a new cache distribution. Linear
    # ramp for `warmup_steps` then cosine for the rest.
    if args.warmup_steps > 0 and args.warmup_steps < total_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, total_steps - args.warmup_steps))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[args.warmup_steps])
        print(f"[build] LR schedule: linear-warmup {args.warmup_steps} steps "
              f"(1e-3x -> 1x) then cosine over {total_steps - args.warmup_steps} steps",
              flush=True)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
        print(f"[build] LR schedule: cosine over {total_steps} steps (no warmup)",
              flush=True)
    loss_kw = dict(w_center=args.w_center, w_depth=args.w_depth,
                    w_dims=args.w_dims, w_rot=args.w_rot)

    # ---------- resume ---------------------------------------------------
    best = 1e9
    history = []
    start_ep = 0
    resume_path = f"{args.out_dir}/resume.pt"
    if args.auto_resume and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=dev, weights_only=False)
        refiner.load_state_dict(ck["refiner"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        best = ck["best"]; history = ck["history"]; start_ep = ck["epoch"] + 1
        print(f"[resume] restored at epoch {start_ep}/{args.epochs} "
              f"(best iou3d={-best:.4f})", flush=True)
    else:
        print("[eval] baseline multi-source:", flush=True)
        ev = evaluate_multi(refiner, val_loaders, dev, weights_by_name)
        print("  unified: " + json.dumps(ev["unified"]), flush=True)
        for n, r in ev["per_source"].items():
            print(f"  {n}: track_c iou3d={r['track_c']['iou3d']:.4f} "
                   f"center={r['track_c']['center_m']:.3f}m "
                   f"rot={r['track_c']['rot_deg']:.2f}deg "
                   f"input iou3d={r['input']['iou3d']:.4f}", flush=True)

    # ---------- train loop ------------------------------------------------
    K = args.batch_trajs
    for ep in range(start_ep, args.epochs):
        refiner.train()
        train_sampler.set_epoch(ep)
        t0 = time.time()
        running, nstep, logs, dlogs, plogs = 0.0, 0, None, None, None
        idxs = list(iter(train_sampler))
        opt.zero_grad()

        # frame-budget grouping (variable-length across CA-1M ~50, Waymo up to ~199, ADT ~51)
        groups, cur, cur_T = [], [], 0
        for i in idxs:
            Ti = int(train_ds.traj_T(i))
            if cur and (len(cur) >= K or cur_T + Ti > args.max_frames_per_batch):
                groups.append(cur); cur, cur_T = [], 0
            cur.append(i); cur_T += Ti
        if cur:
            groups.append(cur)

        for grp in groups:
            packs = [train_ds[i] for i in grp]
            batch = to_dev(collate_trajs(packs), dev)
            frame_mask = None
            if args.mask_frame_p > 0:
                sum_T = batch["hidden"].shape[1]
                frame_mask = (torch.rand(sum_T, device=dev) < args.mask_frame_p)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = refiner.forward_vectorized(batch, frame_mask=frame_mask)
            pred = out["reg"][:, :, 0, :].float()
            valid = batch["gt_center"][:, 2] > 1e-3
            gt_R = quaternion_to_matrix(batch["gt_quat"])
            target, weights_t = encode_targets_batched(
                refiner.coder, batch["gt_center"], batch["gt_dims"],
                batch["gt_quat"], batch["box2d"], batch["K"], batch["input_hw"])
            loss = track_c_loss_from_targets(pred, target, weights_t, gt_R, valid, **loss_kw)
            total = loss["loss"]
            dlog, plog = None, None
            w_pattern = (args.w_rotvel2 + args.w_acc_hinge
                          + args.w_pos_flip + args.w_rot_flip
                          + args.w_l7 + args.w_l8 + args.w_l9a + args.w_l9b)
            if (args.w_vel + args.w_acc + args.w_rotvel) > 0 or w_pattern > 0:
                reg_f = pred[-1]
                off, pcs, pRs = 0, [], []
                for n in batch["sizes"]:
                    sl = slice(off, off + int(n)); off += int(n)
                    dec = refiner.decode_layer(reg_f[sl], batch["box2d"][sl],
                                                batch["K"][sl][0], batch["input_hw"])
                    pcs.append(dec[:, 0:3]); pRs.append(quaternion_to_matrix(dec[:, 6:10]))
                pc_all, pR_all = torch.cat(pcs, 0), torch.cat(pRs, 0)
                if (args.w_vel + args.w_acc + args.w_rotvel) > 0:
                    dlog = derivative_matching_loss(
                        pc_all, pR_all, batch["gt_center"], gt_R,
                        batch["sizes"], valid, w_vel=args.w_vel, w_acc=args.w_acc,
                        w_rotvel=args.w_rotvel)
                    total = total + dlog["loss"]
                if w_pattern > 0:
                    plog = pattern_smoothness_loss(
                        pc_all, pR_all, batch["gt_center"], gt_R,
                        batch["sizes"], valid,
                        w_rotvel2=args.w_rotvel2, rotvel2_form=args.rotvel2_form,
                        w_acc_hinge=args.w_acc_hinge,
                        acc_hinge_margin=args.acc_hinge_margin,
                        w_pos_flip=args.w_pos_flip, w_rot_flip=args.w_rot_flip,
                        flip_lags=flip_lags,
                        pos_delta=(args.pos_delta_mm * 1e-3) ** 2,
                        rot_delta=math.radians(args.rot_delta_deg) ** 2,
                        flip_margin=args.flip_margin,
                        w_l7=args.w_l7, w_l8=args.w_l8,
                        w_l9a=args.w_l9a, w_l9b=args.w_l9b,
                        spec_window=args.spec_window, spec_hop=args.spec_hop,
                        spec_cutoff_period=args.spec_cutoff_period,
                        spec_delta_mm=args.spec_delta_mm,
                        spec_l9b_margin=args.spec_l9b_margin)
                    total = total + plog["loss"]
            # NaN guard: check loss BEFORE backward. If a single batch's loss
            # is NaN/inf, do NOT backward — otherwise NaN gradients corrupt
            # weights and every subsequent step is NaN too (bf16 autocast is
            # susceptible during the initial-domain-shock period). Skipping
            # is a no-op equivalent to picking a different batch; scheduler
            # still advances so LR progression is unchanged.
            if args.skip_bad_steps and not torch.isfinite(total).item():
                if nstep < 10 or nstep % 100 == 0:
                    print(f"[skip] step {nstep}: non-finite loss={float(total):.4g} "
                          f"(center={float(loss['loss_center']):.3g} "
                          f"depth={float(loss['loss_depth']):.3g})", flush=True)
                opt.zero_grad(); sched.step()
                nstep += 1; logs = loss; dlogs = dlog
                plogs = plog if plog is not None else plogs
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in refiner.parameters() if p.requires_grad], 5.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += float(total); nstep += 1; logs = loss; dlogs = dlog
            plogs = plog if plog is not None else plogs
        dt = time.time() - t0
        if refiner.head.temporal_gate is not None:
            gate = torch.stack([g.abs().mean()
                                 for g in refiner.head.temporal_gate[:-1]]).mean().item()
        else:
            gate = 0.0

        print(f"[ep {ep}] train_loss={running/max(nstep,1):.4f} "
              f"(center={float(logs['loss_center']):.3f} depth={float(logs['loss_depth']):.3f} "
              f"dims={float(logs['loss_dims']):.3f} rot_deg={float(logs['loss_rot_deg']):.2f}) "
              f"gate|.|={gate:.4f} lr={sched.get_last_lr()[0]:.2e} {dt:.0f}s "
              f"[{nstep} steps, ~{len(idxs)/max(nstep,1):.1f} tr/step]",
              flush=True)

        rec = {"epoch": ep, "train_loss": running / max(nstep, 1),
                "loss_center": float(logs['loss_center']),
                "loss_depth": float(logs['loss_depth']),
                "loss_dims": float(logs['loss_dims']),
                "loss_rot_deg": float(logs['loss_rot_deg']),
                "gate": gate, "lr": sched.get_last_lr()[0], "epoch_sec": dt}
        if dlogs is not None:
            rec.update(d_center_vel=float(dlogs['d_center_vel']),
                        d_center_acc=float(dlogs['d_center_acc']),
                        d_rot_vel_deg=float(dlogs['d_rot_vel_deg']))
        if plogs is not None:
            rec.update({k: float(v) for k, v in plogs.items() if k != "loss"})

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            ev = evaluate_multi(refiner, val_loaders, dev, weights_by_name)
            u = ev["unified"]
            per_str = " | ".join(
                f"{n}:{r['track_c']['iou3d']:.3f}" for n, r in ev["per_source"].items())
            print(f"[ep {ep}] EVAL unified iou3d={u['iou3d']:.4f} "
                  f"center={u['center_m']:.3f}m rot={u['rot_deg']:.2f}deg  ||  {per_str}",
                  flush=True)
            for n, r in ev["per_source"].items():
                tc = r["track_c"]; inp = r["input"]
                print(f"[ep {ep}]   {n}: track_c iou3d={tc['iou3d']:.4f} "
                      f"center={tc['center_m']:.3f}m dims={tc['dims_m']:.3f}m "
                      f"rot={tc['rot_deg']:.2f}deg | input iou3d={inp['iou3d']:.4f} "
                      f"center={inp['center_m']:.3f}m rot={inp['rot_deg']:.2f}deg "
                      f"nfr={r['_n_frames']}", flush=True)
            rec["eval"] = ev
            score = -u["iou3d"]
            if score < best:
                best = score
                torch.save({"refiner": refiner.state_dict(), "args": vars(args),
                             "epoch": ep, "eval": ev}, f"{args.out_dir}/best.pt")
                print(f"[ep {ep}] saved best (unified iou3d={u['iou3d']:.4f})",
                      flush=True)
        history.append(rec)
        with open(f"{args.out_dir}/history.json", "w") as f:
            json.dump(history, f, indent=1)
        tmp = f"{args.out_dir}/resume.pt.tmp"
        torch.save({"refiner": refiner.state_dict(), "opt": opt.state_dict(),
                     "sched": sched.state_dict(), "epoch": ep, "best": best,
                     "history": history, "args": vars(args)}, tmp)
        os.replace(tmp, f"{args.out_dir}/resume.pt")
    torch.save({"refiner": refiner.state_dict(), "args": vars(args)},
                f"{args.out_dir}/last.pt")
    print("[train] done", flush=True)


if __name__ == "__main__":
    main()
