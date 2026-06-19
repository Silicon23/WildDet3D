"""Saturation-stratified temporal eval — does temporal ON beat OFF in the
per-frame-UNDERdetermined cells (small / occluded / weak-prior)?

The thesis: temporal helps where a single frame can't determine the box. The
scout's saturation_tags.jsonl gives per-track axes — median_mask_px (size),
mask_fill_ratio (occlusion: mask px / projected GT 2D box area; low=occluded),
prior_iou3d_scaled_median (prior quality). We run the temporal OFF and ON models
on val tracks, then report OFF vs ON IoU3D split by tertiles of each axis. If
temporal ever earns its keep, the ON-OFF delta should be positive in the
small / low-fill-ratio / low-prior-IoU tertiles even if the aggregate is null.

Usage:
  python -m wilddet3d.track_c.saturation_strat_eval --cache_dir <c> \
    --off_ckpt <off> --on_ckpt <on> --val_split_file <split> \
    --categories vehicle --pairing_index <idx> --tags <saturation_tags.jsonl>
"""
import sys, argparse, json, os, numpy as np, torch
WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
sys.path.insert(0, WD)
from wilddet3d.ops.iou_3d_safe import batch_box3d_iou
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import CachedTrackCDataset, list_cached_trajectories, split_by_video


def run_model(ckpt, on, ds, dev):
    r = TrackCRefiner(reg_residual_from_prior=False, use_temporal_modules=on,
                      use_temporal_kv_norm=on, temporal_multi_token=on).to(dev)
    r.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["refiner"], strict=True)
    r.eval()
    per_track = []
    with torch.no_grad():
        for i in range(len(ds)):
            p = ds[i]; g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = r(hidden_states=g["hidden"].float(), ray_embeddings=g["ray"].float(),
                      depth_latents=g["depth"].float(), pred_box_2d=g["box2d"], intrinsics=g["K"],
                      box_repr=g["box_repr"], timestamps=g["ts"], measured_mask=g["measured"],
                      input_hw=g["input_hw"])
            dec = r.decode_layer(o["reg"][-1, :, 0, :].float(), g["box2d"], g["K"], g["input_hw"]).cpu()
            gt = torch.cat([p["gt_center"].float(), p["gt_dims"].float(), p["gt_quat"].float()], -1)
            per_track.append(float(batch_box3d_iou(dec, gt).mean()))  # mean per-frame IoU for the track
    return np.array(per_track)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--off_ckpt", required=True)
    ap.add_argument("--on_ckpt", required=True)
    ap.add_argument("--val_split_file", required=True)
    ap.add_argument("--categories", default="")
    ap.add_argument("--pairing_index", default="")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    tags = {}
    for line in open(a.tags):
        d = json.loads(line); tags[f"{d['seg']}__{d['track_id']}"] = d

    paths = list_cached_trajectories(a.cache_dir)
    if a.categories and a.pairing_index:
        keep = set(c.strip() for c in a.categories.split(",") if c.strip())
        cat = {}
        for line in open(a.pairing_index):
            d = json.loads(line); cat[f"{d['seg']}__{d['track_id']}"] = d["category"]
        paths = [p for p in paths if cat.get(os.path.basename(p)[:-3]) in keep]
    _, val_paths, _ = split_by_video(paths, 0.12, split_file=a.val_split_file)
    ds = CachedTrackCDataset(a.cache_dir, val_paths, preload=True)
    keys = [os.path.basename(val_paths[i])[:-3] for i in range(len(ds))]

    off = run_model(a.off_ckpt, False, ds, dev)
    on = run_model(a.on_ckpt, True, ds, dev)

    px = np.array([tags.get(k, {}).get("median_mask_px", np.nan) for k in keys], float)
    fr = np.array([tags.get(k, {}).get("mask_fill_ratio", np.nan) for k in keys], float)
    pi = np.array([tags.get(k, {}).get("prior_iou3d_scaled_median", np.nan) for k in keys], float)

    print(f"\n=== SATURATION-STRATIFIED temporal eval ({len(ds)} val tracks) ===")
    print(f"AGGREGATE: OFF iou {off.mean():.4f}  ON iou {on.mean():.4f}  dIoU(ON-OFF) {on.mean()-off.mean():+.4f}")
    for name, ax, lohi in [("mask_px(size)", px, "low=small"),
                           ("fill_ratio(occl)", fr, "low=occluded"),
                           ("prior_iou", pi, "low=weak-prior")]:
        ok = ~np.isnan(ax)
        if ok.sum() < 6:
            print(f"\n{name}: too few tagged tracks"); continue
        q1, q2 = np.nanpercentile(ax[ok], [33, 67])
        print(f"\n{name} ({lohi}); tertile cuts {q1:.1f}, {q2:.1f}")
        print(f"  {'tertile':10} {'n':>3} {'OFF':>8} {'ON':>8} {'dIoU':>8}")
        for lab, m in [("low", ax <= q1), ("mid", (ax > q1) & (ax <= q2)), ("high", ax > q2)]:
            m = m & ok
            if m.sum() == 0: continue
            print(f"  {lab:10} {int(m.sum()):3d} {off[m].mean():8.4f} {on[m].mean():8.4f} {on[m].mean()-off[m].mean():+8.4f}")
    print("\n(temporal-necessary cells = LOW size / LOW fill_ratio / LOW prior_iou; "
          "look for dIoU>0 there beyond the aggregate)")


if __name__ == "__main__":
    main()
