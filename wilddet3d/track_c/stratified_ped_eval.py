"""Stratified near/far pedestrian eval — the test the ped re-cut was designed for.

The temporal hypothesis on peds is specific: neighbors carry confident-near pose
INTO the ambiguous far/occluded frames. The aggregate IoU averages confident-near
+ hopeless-far and hides that. Here we split each val frame by the ped index's
per-frame `near_field` (<15 m) flag and report temporal OFF vs ON SEPARATELY on
near vs far frames. If temporal helps, it shows on the FAR stratum even when the
aggregate is flat.

Usage:
  python -m wilddet3d.track_c.stratified_ped_eval \
    --cache_dir <ped cache> --ped_index <ped jsonl> --val_split_file <ped val> \
    --off_ckpt <Pk off best.pt> --on_ckpt <Pk on best.pt>
"""
import sys, argparse, json, os, numpy as np, torch
WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
sys.path.insert(0, WD)
from vis4d.op.geometry.rotation import quaternion_to_matrix
from wilddet3d.ops.iou_3d_safe import batch_box3d_iou
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import CachedTrackCDataset, list_cached_trajectories, split_by_video


def near_map(ped_index):
    """{(seg,track): {step1_index: near_field bool}}."""
    m = {}
    for line in open(ped_index):
        d = json.loads(line)
        m[(d["seg"], d["track_id"])] = {int(f["step1_index"]): bool(f["near_field"])
                                        for f in d["frames"]}
    return m


def run_model(ckpt, on, ds, dev):
    r = TrackCRefiner(reg_residual_from_prior=False,
                      use_temporal_modules=on, use_temporal_kv_norm=on,
                      temporal_multi_token=on).to(dev)
    r.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["refiner"], strict=True)
    r.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(ds)):
            p = ds[i]; g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = r(hidden_states=g["hidden"].float(), ray_embeddings=g["ray"].float(),
                      depth_latents=g["depth"].float(), pred_box_2d=g["box2d"], intrinsics=g["K"],
                      box_repr=g["box_repr"], timestamps=g["ts"], measured_mask=g["measured"],
                      input_hw=g["input_hw"])
            dec = r.decode_layer(o["reg"][-1, :, 0, :].float(), g["box2d"], g["K"], g["input_hw"])
            preds.append(dec.cpu())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ped_index", required=True)
    ap.add_argument("--val_split_file", required=True)
    ap.add_argument("--off_ckpt", required=True)
    ap.add_argument("--on_ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device

    nmap = near_map(a.ped_index)
    paths = list_cached_trajectories(a.cache_dir)
    _, val_paths, _ = split_by_video(paths, 0.15, split_file=a.val_split_file)
    ds = CachedTrackCDataset(a.cache_dir, val_paths, preload=True)

    off = run_model(a.off_ckpt, False, ds, dev)
    on = run_model(a.on_ckpt, True, ds, dev)

    # accumulate per-frame IoU + center err, stratified by near_field
    agg = {m: {"near": {"iou": [], "ce": []}, "far": {"iou": [], "ce": []}}
           for m in ("off", "on")}
    for i in range(len(ds)):
        p = ds[i]
        seg, trk = p["video_id"], p["object_id"]
        fidx = ds.traj_cache[i]["frame_index"].tolist()
        nm = nmap.get((seg, trk), {})
        gt = torch.cat([p["gt_center"].float(), p["gt_dims"].float(),
                        p["gt_quat"].float()], -1)                       # [T,10]
        for name, preds in (("off", off), ("on", on)):
            dec = preds[i]                                               # [T,10] center,dims,quat
            iou = batch_box3d_iou(dec, gt).numpy()                      # [T]
            ce = (dec[:, 0:3] - gt[:, 0:3]).norm(dim=-1).numpy()
            for t, fi in enumerate(fidx):
                strat = "near" if nm.get(int(fi), False) else "far"
                agg[name][strat]["iou"].append(float(iou[t]))
                agg[name][strat]["ce"].append(float(ce[t]))

    print(f"\n=== STRATIFIED PED EVAL (val {len(ds)} trajs) ===")
    print(f"{'stratum':6} {'n':>6}  {'OFF iou':>8} {'ON iou':>8} {'dIoU':>7}  "
          f"{'OFF ce':>7} {'ON ce':>7} {'dCe':>7}")
    for strat in ("near", "far"):
        n = len(agg["off"][strat]["iou"])
        oi, ni = np.mean(agg["off"][strat]["iou"]), np.mean(agg["on"][strat]["iou"])
        oc, nc = np.mean(agg["off"][strat]["ce"]), np.mean(agg["on"][strat]["ce"])
        print(f"{strat:6} {n:6d}  {oi:8.4f} {ni:8.4f} {ni-oi:+7.4f}  "
              f"{oc:7.3f} {nc:7.3f} {nc-oc:+7.3f}")
    print("\n(temporal helps far/occluded => dIoU>0 / dCe<0 on the FAR row, the designed signal)")


if __name__ == "__main__":
    main()
