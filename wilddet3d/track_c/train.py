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
from wilddet3d.track_c.losses import (
    build_targets,
    derivative_matching_loss,
    encode_targets_batched,
    track_c_loss_from_targets,
)
from wilddet3d.track_c.dataset import (
    CachedTrackCDataset,
    VideoGroupedSampler,
    collate_trajs,
    list_cached_trajectories,
    split_by_video,
)
from wilddet3d.track_c.losses import geodesic_rotation_loss
from wilddet3d.track_c.smoothness_losses import pattern_smoothness_loss


def to_dev(pack, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in pack.items()}


def forward_loss(refiner, pack, loss_kw):
    # bf16 autocast for the heavy 3D-head/traj-encoder matmuls; the temporal-box
    # geometry stays fp32 (forced inside refiner.forward) and the loss runs fp32.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = refiner(
            hidden_states=pack["hidden"], ray_embeddings=pack["ray"],
            depth_latents=pack["depth"], pred_box_2d=pack["box2d"],
            intrinsics=pack["K"], box_repr=pack["box_repr"],
            timestamps=pack["ts"], measured_mask=pack["measured"],
            input_hw=pack["input_hw"],
        )
    gt_R = quaternion_to_matrix(pack["gt_quat"])
    loss = track_c_loss(
        out["reg"].float(), refiner.coder, pack["gt_center"], pack["gt_dims"],
        pack["gt_quat"], gt_R, pack["box2d"], pack["K"], pack["input_hw"],
        valid=(pack["gt_center"][:, 2] > 1e-3), **loss_kw,
    )
    return out, loss


def forward_loss_batch(refiner, packs, loss_kw):
    """Batched training step: one head forward over many trajectories' frames."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = refiner.forward_batch(packs)
    pred = out["reg"][:, :, 0, :].float()                  # [L, sum_T, 12]
    targets, weights, gtRs, valids = [], [], [], []
    for p in packs:
        tgt, w = build_targets(refiner.coder, p["gt_center"], p["gt_dims"],
                               p["gt_quat"], p["box2d"], p["K"], p["input_hw"])
        targets.append(tgt)
        weights.append(w)
        gtRs.append(quaternion_to_matrix(p["gt_quat"]))
        valids.append(p["gt_center"][:, 2] > 1e-3)
    loss = track_c_loss_from_targets(
        pred, torch.cat(targets, 0), torch.cat(weights, 0),
        torch.cat(gtRs, 0), torch.cat(valids, 0), **loss_kw,
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
        # trajectory-encoder box (only present when modules are active)
        if out.get("traj_box_out") is not None:
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
    ap.add_argument("--accum", type=int, default=8, help="(unused in batched loop)")
    ap.add_argument("--batch_trajs", type=int, default=8,
                    help="MAX trajectories concatenated per batched head forward / step")
    ap.add_argument("--max_frames_per_batch", type=int, default=360,
                    help="cap total frames/batch (head forward concats all frames; "
                         "guards OOM on long variable-length tracks). 360 = CA-1M's "
                         "proven-safe max, so CA-1M batches stay K-limited (unchanged)")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0,
                    help="torch manual seed (model init + RNG split); vary for variance runs")
    ap.add_argument("--val_split_file", default=None,
                    help="path to a {val_video_ids:[...]} json; default=CA-1M canonical. "
                         "Pass a Waymo/ADT split here, or '' to force RNG split by val_frac/seed")
    ap.add_argument("--categories", default="",
                    help="comma list to keep (e.g. 'vehicle'); needs --pairing_index. empty=all")
    ap.add_argument("--pairing_index", default="",
                    help="pairing_index.jsonl to map <seg>__<track> -> category for --categories")
    ap.add_argument("--gate_lr_mult", type=float, default=20.0,
                    help="LR multiplier for the new temporal modules (gate/prompt)")
    ap.add_argument("--reg_residual_from_prior", type=int, default=1)
    ap.add_argument("--no_temporal", type=int, default=0,
                    help="ablation: freeze the temporal-prompt gate at 0 (no "
                         "temporal cross-attention; pure per-frame image grounding)")
    ap.add_argument("--no_traj_encoder", type=int, default=0,
                    help="v11: physically remove TrajectoryEncoder + prompt_temporal "
                         "(use_temporal_modules=False). Modules are not allocated.")
    ap.add_argument("--use_layer_bias", type=int, default=0,
                    help="v11_with_bias: add a per-layer learned bias (256-d) to "
                         "the head, replacing the collapsed temporal feature cheaply.")
    ap.add_argument("--temporal_kv_norm", type=int, default=0,
                    help="scale-fix: LayerNorm the projected temporal KV before the "
                         "cross-attn (caps the ~100x KV-scale drift of the from-scratch branch).")
    ap.add_argument("--warm_start_temporal", type=int, default=0,
                    help="scale-fix: init project_temporal/prompt_temporal from the "
                         "pretrained depth branch instead of from scratch.")
    ap.add_argument("--temporal_multi_token", type=int, default=0,
                    help="Flavor 2: per-frame query attends the whole within-object "
                         "timeline of temporal tokens (timestamp PE on q+k, block-diagonal "
                         "mask) instead of a single token. Breaks the single-key degeneracy.")
    ap.add_argument("--temporal_block", default="prompt3d",
                    choices=["prompt3d", "xattn_only"],
                    help="xattn_only = pure cross-attn temporal block (no self-attn/FFN "
                         "query-only shortcut; the 2026-07-03 fix). prompt3d = original.")
    ap.add_argument("--mask_frame_p", type=float, default=0.0,
                    help="masked-frame objective: fraction of frames per batch whose "
                         "visual evidence (hidden state + depth latents) is withheld "
                         "during training; their box is still supervised, forcing the "
                         "temporal cross-attn to read neighbors' tokens. 0 = off.")
    ap.add_argument("--w_center", type=float, default=1.0)
    ap.add_argument("--w_depth", type=float, default=1.0)
    ap.add_argument("--w_dims", type=float, default=1.0)
    ap.add_argument("--w_rot", type=float, default=1.0)
    ap.add_argument("--w_vel", type=float, default=0.0,
                    help="GT-derivative match: center velocity (1st diff)")
    ap.add_argument("--w_acc", type=float, default=0.0,
                    help="GT-derivative match: center acceleration (2nd diff)")
    ap.add_argument("--w_rotvel", type=float, default=0.0,
                    help="GT-derivative match: angular velocity (deg, symmetry-aware) — the main smoothness term")
    # --- pattern smoothness terms (trajectory_smoothness_losses.md, first wave) ---
    ap.add_argument("--w_rotvel2", type=float, default=0.0,
                    help="R2: UNFOLDED relative-rotation matching (chordal). Catches "
                         "axis wobble + mid-track flips the folded w_rotvel cannot see.")
    ap.add_argument("--rotvel2_form", default="chordal", choices=["chordal", "geodesic"])
    ap.add_argument("--w_acc_hinge", type=float, default=0.0,
                    help="L3h: ReLU(|acc_pred|-|acc_gt|-margin), linear in jitter amplitude")
    ap.add_argument("--acc_hinge_margin", type=float, default=0.0)
    ap.add_argument("--w_pos_flip", type=float, default=0.0,
                    help="L5/L6: lag-k flip loss on center velocity deviation (scale-invariant)")
    ap.add_argument("--w_rot_flip", type=float, default=0.0,
                    help="L5/L6 on gated so(3) log-increment deviation")
    ap.add_argument("--flip_lags", default="1,2", help="comma lags for the flip losses")
    ap.add_argument("--pos_delta_mm", type=float, default=10.0,
                    help="sqrt(delta) for position flip loss, mm/frame (noise-floor gate)")
    ap.add_argument("--rot_delta_deg", type=float, default=0.05,
                    help="sqrt(delta) for rotation flip loss, deg/frame")
    ap.add_argument("--flip_margin", type=float, default=0.2)
    # Spectral family (doc §5) — operate on velocity deviation d_t = v_pred - v_gt
    ap.add_argument("--w_l7", type=float, default=0.0,
                    help="L7 self-normalized HF energy fraction (watch drift gaming)")
    ap.add_argument("--w_l8", type=float, default=0.0,
                    help="L8 soft band edge (Hann ramp over one octave above k_c)")
    ap.add_argument("--w_l9a", type=float, default=0.0,
                    help="L9a GT-energy denominator (structural anti-gaming variant of L7)")
    ap.add_argument("--w_l9b", type=float, default=0.0,
                    help="L9b GT-referenced HF-ratio hinge on predicted velocity")
    ap.add_argument("--spec_window", type=int, default=16,
                    help="spectral window length (frames); 50% overlap via spec_hop")
    ap.add_argument("--spec_hop", type=int, default=8)
    ap.add_argument("--spec_cutoff_period", type=float, default=4.0,
                    help="period (frames) at high-band cutoff; k_c = window/period")
    ap.add_argument("--spec_delta_mm", type=float, default=50.0,
                    help="velocity-deviation noise-floor stabilizer, mm/frame")
    ap.add_argument("--spec_l9b_margin", type=float, default=0.05,
                    help="L9b hinge: penalize when rho(v_pred) > rho(v_gt) + margin")
    ap.add_argument("--init_from", default="",
                    help="warm-start the refiner from a previous run's best.pt "
                         "(arch flags must match); use with few-epoch fine-tunes")
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--auto_resume", type=int, default=1,
                    help="if <out_dir>/resume.pt exists, restore full state "
                         "(model+opt+sched+epoch+best+history) and continue — "
                         "makes container/session restarts cheap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device

    torch.manual_seed(args.seed)
    paths = list_cached_trajectories(args.cache_dir)
    if args.categories and args.pairing_index:
        import json as _json
        keep_cats = set(c.strip() for c in args.categories.split(",") if c.strip())
        cat_of = {}
        for line in open(args.pairing_index):
            d = _json.loads(line)
            cat_of[f"{d['seg']}__{d['track_id']}"] = d["category"]
        paths = [p for p in paths
                 if cat_of.get(os.path.basename(p)[:-3]) in keep_cats]
        print(f"[data] category filter {keep_cats}: {len(paths)} trajs kept", flush=True)
    if args.limit:
        paths = paths[:args.limit]
    split_kw = {} if args.val_split_file is None else {"split_file": args.val_split_file}
    train_paths, val_paths, val_vids = split_by_video(
        paths, args.val_frac, seed=args.seed, **split_kw)
    print(f"[data] {len(paths)} trajs -> train {len(train_paths)} / val {len(val_paths)} "
          f"({len(val_vids)} val videos)", flush=True)

    print("[data] preloading frames + traj features into RAM ...", flush=True)
    train_ds = CachedTrackCDataset(args.cache_dir, train_paths, preload=True)
    val_ds = CachedTrackCDataset(args.cache_dir, val_paths, preload=True)
    train_sampler = VideoGroupedSampler(train_ds, shuffle=True)
    # num_workers=0: keep the per-video frames-file LRU in one process so the
    # video-grouped sampler loads each ~90MB frames file once per epoch. With
    # workers, round-robin index distribution defeats that locality.
    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=0)

    refiner = TrackCRefiner(
        reg_residual_from_prior=bool(args.reg_residual_from_prior),
        use_temporal_modules=not bool(args.no_traj_encoder),
        use_layer_bias=bool(args.use_layer_bias),
        use_temporal_kv_norm=bool(args.temporal_kv_norm),
        temporal_multi_token=bool(args.temporal_multi_token),
        temporal_block=args.temporal_block,
    ).to(dev)
    info = refiner.load_pretrained_head(args.ckpt)
    print(f"[build] head load: {info['loaded']} tensors, new={len(info['missing'])}", flush=True)
    refiner.finalize_init()
    if args.warm_start_temporal:
        refiner.warm_start_temporal_from_depth()
        print("[build] warm-started temporal branch from pretrained depth branch", flush=True)
    if args.no_temporal and refiner.head.temporal_gate is not None:
        for g in refiner.head.temporal_gate:
            g.data.zero_()
            g.requires_grad_(False)
        print("[build] no_temporal: temporal-prompt gates frozen at 0 (ablation)", flush=True)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)["refiner"]
        # strict=False so we can warm-start a temporal-ON refiner from a temporal-OFF
        # checkpoint (or vice-versa): the source ckpt lacks the temporal modules
        # (traj_encoder, project_temporal, prompt_temporal, temporal_gate,
        # temporal_kv_norm) that the destination has, or has them but they're
        # absent from the destination. Report the delta so silent mismatches are
        # visible in the log.
        missing, unexpected = refiner.load_state_dict(sd, strict=False)
        print(f"[build] warm-started full refiner from {args.init_from} "
              f"(loaded, missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
        if missing:
            print(f"[build]   missing keys sample: {missing[:3]}{'...' if len(missing) > 3 else ''}", flush=True)
        if unexpected:
            print(f"[build]   unexpected keys sample: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}", flush=True)

    # Higher LR for the new temporal modules (esp. the zero-init LayerScale gate)
    flip_lags = tuple(int(x) for x in str(args.flip_lags).split(",") if x.strip())

    # so the image-grounded temporal-prompt pathway activates from a cold start,
    # while preserving identity-at-init (gate starts at exactly 0).
    temporal_params, base_params = [], []
    for name, p in refiner.named_parameters():
        if not p.requires_grad:
            continue
        if ("temporal_gate" in name or "prompt_temporal" in name
                or "project_temporal" in name or "temporal_kv_norm" in name
                or "layer_bias" in name):
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
    total_steps = args.epochs * max(1, len(train_paths) // args.batch_trajs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))
    loss_kw = dict(w_center=args.w_center, w_depth=args.w_depth,
                   w_dims=args.w_dims, w_rot=args.w_rot)

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
        print(f"[resume] restored from {resume_path}: continuing at epoch "
              f"{start_ep}/{args.epochs} (best iou3d={-best:.4f})", flush=True)
    else:
        print("[eval] baseline (init / before training):", flush=True)
        ev = evaluate(refiner, val_loader, dev)
        print("  " + json.dumps(ev), flush=True)

    K = args.batch_trajs
    for ep in range(start_ep, args.epochs):
        refiner.train()
        train_sampler.set_epoch(ep)
        t0 = time.time()
        running, nstep, logs, dlogs, plogs = 0.0, 0, None, None, None
        idxs = list(iter(train_sampler))
        opt.zero_grad()
        # Group into batches by a FRAME budget, not a fixed traj count: the head
        # forward concatenates all frames in a batch, so memory scales with total
        # frames. Variable-length tracks (Waymo up to ~199 vs CA-1M ~60) OOM at a
        # fixed K. Cap total frames/batch (>=1 traj even if it alone exceeds).
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
            # Masked-frame objective: withhold visual evidence for a random
            # subset of frames (train-time only); their boxes stay supervised.
            frame_mask = None
            if args.mask_frame_p > 0:
                sum_T = batch["hidden"].shape[1]
                frame_mask = (torch.rand(sum_T, device=dev) < args.mask_frame_p)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = refiner.forward_vectorized(batch, frame_mask=frame_mask)
            pred = out["reg"][:, :, 0, :].float()
            valid = batch["gt_center"][:, 2] > 1e-3
            gt_R = quaternion_to_matrix(batch["gt_quat"])
            target, weights = encode_targets_batched(
                refiner.coder, batch["gt_center"], batch["gt_dims"],
                batch["gt_quat"], batch["box2d"], batch["K"], batch["input_hw"])
            loss = track_c_loss_from_targets(pred, target, weights, gt_R, valid, **loss_kw)
            total = loss["loss"]
            dlog, plog = None, None
            w_pattern = (args.w_rotvel2 + args.w_acc_hinge
                         + args.w_pos_flip + args.w_rot_flip
                         + args.w_l7 + args.w_l8 + args.w_l9a + args.w_l9b)
            if (args.w_vel + args.w_acc + args.w_rotvel) > 0 or w_pattern > 0:
                # decode final-layer boxes per trajectory (single K each) in fp32
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
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in refiner.parameters() if p.requires_grad], 5.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += float(total); nstep += 1; logs = loss; dlogs = dlog
            plogs = plog if plog is not None else plogs
        dt = time.time() - t0
        # mean over the USED prediction layers (exclude the last clone, which the
        # 6-decoder-layer forward never invokes, so it stays at its zero init).
        if refiner.head.temporal_gate is not None:
            gate = torch.stack([g.abs().mean()
                                for g in refiner.head.temporal_gate[:-1]]).mean().item()
        else:
            gate = 0.0
        dstr = ""
        if dlogs is not None:
            dstr = (f" deriv(cvel={float(dlogs['d_center_vel']):.3f} "
                    f"cacc={float(dlogs['d_center_acc']):.3f} "
                    f"rotvel_deg={float(dlogs['d_rot_vel_deg']):.2f})")
        if plogs is not None:
            keys = [k for k in ("loss_rotvel2", "loss_acc_hinge", "loss_pos_flip",
                                "loss_rot_flip") if k in plogs]
            mons = [k for k in ("pos_mon_lag1_cos", "pos_mon_lag1_active",
                                "rot_mon_lag1_cos", "rot_mon_gate_pass",
                                "mon_flip_rate", "mon_acc_hinge_active")
                    if k in plogs]
            dstr += (" pat(" + " ".join(f"{k.replace('loss_','')}={float(plogs[k]):.4f}"
                                        for k in keys)
                     + " | " + " ".join(f"{k.replace('_mon','')}={float(plogs[k]):.3f}"
                                        for k in mons) + ")")
        print(f"[ep {ep}] train_loss={running/max(nstep,1):.4f} "
              f"(center={float(logs['loss_center']):.3f} depth={float(logs['loss_depth']):.3f} "
              f"dims={float(logs['loss_dims']):.3f} rot_deg={float(logs['loss_rot_deg']):.2f}){dstr} "
              f"gate|.|={gate:.4f} lr={sched.get_last_lr()[0]:.2e} {dt:.0f}s", flush=True)

        rec = {"epoch": ep, "train_loss": running / max(nstep, 1),
               "loss_center": float(logs['loss_center']), "loss_depth": float(logs['loss_depth']),
               "loss_dims": float(logs['loss_dims']), "loss_rot_deg": float(logs['loss_rot_deg']),
               "gate": gate, "lr": sched.get_last_lr()[0], "epoch_sec": dt}
        if dlogs is not None:
            rec.update(d_center_vel=float(dlogs['d_center_vel']),
                       d_center_acc=float(dlogs['d_center_acc']),
                       d_rot_vel_deg=float(dlogs['d_rot_vel_deg']))
        if plogs is not None:
            rec.update({k: float(v) for k, v in plogs.items() if k != "loss"})
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            ev = evaluate(refiner, val_loader, dev)
            tc = ev["track_c"]; inp = ev["input"]
            print(f"[ep {ep}] EVAL track_c: iou3d={tc['iou3d']:.3f} center={tc['center_m']:.3f}m "
                  f"dims={tc['dims_m']:.3f}m rot={tc['rot_deg']:.2f}deg | input(step4): "
                  f"iou3d={inp['iou3d']:.3f} center={inp['center_m']:.3f}m dims={inp['dims_m']:.3f}m "
                  f"rot={inp['rot_deg']:.2f}deg | nfr={ev['_n_frames']}",
                  flush=True)
            rec["eval"] = ev
            score = -tc["iou3d"]  # maximize 3D IoU (lower score = better)
            if score < best:
                best = score
                torch.save({"refiner": refiner.state_dict(), "args": vars(args),
                            "epoch": ep, "eval": ev}, f"{args.out_dir}/best.pt")
                print(f"[ep {ep}] saved best (iou3d={tc['iou3d']:.4f})", flush=True)
        history.append(rec)
        with open(f"{args.out_dir}/history.json", "w") as f:
            json.dump(history, f, indent=1)
        # full-state checkpoint for cheap resume after container/session restart
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
