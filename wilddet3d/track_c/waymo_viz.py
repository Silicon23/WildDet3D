"""Render OFF-vs-ON Track C 3D boxes on Waymo frames (temporal-channel comparison).

For a segment, runs the temporal-OFF and temporal-ON checkpoints on its cached
tracks, decodes per-frame camera boxes, and overlays on the RGB frames:
  GREEN = GT, RED = temporal OFF, BLUE = temporal ON.
So the difference the temporal channel makes (or doesn't) is directly visible.

Usage:
  python -m wilddet3d.track_c.waymo_viz --seg <SEG> --cache_dir <cache> \
    --off_ckpt <off best.pt> --on_ckpt <on best.pt> --out <dir> [--fps 10]
"""
import sys, argparse, glob, os, json
import numpy as np, torch, cv2
WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
sys.path.insert(0, WD)
from vis4d.op.geometry.rotation import quaternion_to_matrix
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import CachedTrackCDataset

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def corners_cam(center, dims, R):
    """center[3], dims[3]=lhw, R[3,3] -> 8 corners in camera frame [8,3]."""
    l, w, h = dims
    s = np.array([[sx*l/2, sy*w/2, sz*h/2] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])
    # reorder to EDGES convention: (-,-,-),(+,-,-),(+,+,-),(-,+,-),(-,-,+),(+,-,+),(+,+,+),(-,+,+)
    order = [0,4,6,2,1,5,7,3]
    s = s[order]
    return (R @ s.T).T + center


def project(corners, K):
    """[8,3] cam -> [8,2] px (NaN if behind camera)."""
    out = np.full((8, 2), np.nan)
    for i, c in enumerate(corners):
        if c[2] > 0.1:
            p = K @ c
            out[i] = p[:2] / p[2]
    return out


def draw_box(img, px, color, thick=2):
    h, w = img.shape[:2]
    for a, b in EDGES:
        if np.isnan(px[a]).any() or np.isnan(px[b]).any():
            continue
        pa, pb = px[a].astype(int), px[b].astype(int)
        if (max(abs(pa)) < 1e5 and max(abs(pb)) < 1e5):
            cv2.line(img, tuple(pa), tuple(pb), color, thick, cv2.LINE_AA)


def infer(ckpt, on, ds, idxs, dev):
    r = TrackCRefiner(reg_residual_from_prior=False, use_temporal_modules=on,
                      use_temporal_kv_norm=on, temporal_multi_token=on).to(dev)
    r.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["refiner"], strict=True)
    r.eval()
    res = {}
    with torch.no_grad():
        for i in idxs:
            p = ds[i]; g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = r(hidden_states=g["hidden"].float(), ray_embeddings=g["ray"].float(),
                      depth_latents=g["depth"].float(), pred_box_2d=g["box2d"], intrinsics=g["K"],
                      box_repr=g["box_repr"], timestamps=g["ts"], measured_mask=g["measured"],
                      input_hw=g["input_hw"])
            dec = r.decode_layer(o["reg"][-1, :, 0, :].float(), g["box2d"], g["K"], g["input_hw"])
            res[i] = dec.cpu().numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--off_ckpt", required=True)
    ap.add_argument("--on_ckpt", required=True)
    ap.add_argument("--outputs", default="/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--layout", default="sidebyside", choices=["sidebyside", "overlay"],
                    help="sidebyside = GT+OFF | GT+ON panels (no occlusion); overlay = all on one")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    all_paths = sorted(glob.glob(f"{a.cache_dir}/traj/{a.seg}__*.pt"))
    assert all_paths, f"no cached tracks for {a.seg}"
    ds = CachedTrackCDataset(a.cache_dir, all_paths, preload=True)
    idxs = list(range(len(ds)))
    off = infer(a.off_ckpt, False, ds, idxs, dev)
    on = infer(a.on_ckpt, True, ds, idxs, dev)

    K = np.load(f"{a.outputs}/step1_waymo/{a.seg}/intrinsics.npy")
    per_frame_K = K.ndim == 3
    frames_dir = f"{a.outputs}/step1_waymo/{a.seg}/frames"
    # build per-frame: list of (gt10, off10, on10)
    by_frame = {}
    for i in idxs:
        p = ds[i]; fidx = ds.traj_cache[i]["frame_index"].tolist()
        gt = torch.cat([p["gt_center"].float(), p["gt_dims"].float(), p["gt_quat"].float()], -1).numpy()
        for t, fi in enumerate(fidx):
            by_frame.setdefault(int(fi), []).append((gt[t], off[i][t], on[i][t]))

    frame_ids = sorted(by_frame.keys())
    im0 = cv2.imread(f"{frames_dir}/{frame_ids[0]:06d}.jpg")
    H, W = im0.shape[:2]
    GREEN, RED, BLUE = (0,200,0), (0,0,230), (230,80,0)
    side = (a.layout == "sidebyside")
    # side-by-side: LEFT = GT+OFF, RIGHT = GT+ON, so neither prediction occludes the
    # other (the overlay's blue-on-top made ON look better than it is).
    out_w = W * 2 if side else W
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (out_w, H))

    def b10(x):
        return corners_cam(x[0:3], x[3:6], quaternion_to_matrix(torch.tensor(x[6:10])).numpy())

    for fi in frame_ids:
        img = cv2.imread(f"{frames_dir}/{fi:06d}.jpg")
        if img is None:
            continue
        Ki = (K[fi] if per_frame_K else K)
        if side:
            L, Rp = img.copy(), img.copy()
            for gt, of, oncol in by_frame[fi]:
                draw_box(L, project(b10(gt), Ki), GREEN, 2)
                draw_box(L, project(b10(of), Ki), RED, 2)
                draw_box(Rp, project(b10(gt), Ki), GREEN, 2)
                draw_box(Rp, project(b10(oncol), Ki), BLUE, 2)
            cv2.putText(L, "temporal OFF (red) vs GT (green)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(Rp, "temporal ON (blue) vs GT (green)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            vw.write(cv2.hconcat([L, Rp]))
        else:
            for gt, of, oncol in by_frame[fi]:
                draw_box(img, project(b10(gt), Ki), GREEN, 2)
                draw_box(img, project(b10(of), Ki), RED, 2)
                draw_box(img, project(b10(oncol), Ki), BLUE, 2)
            cv2.putText(img, "GT=green  OFF=red  ON=blue", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
            vw.write(img)
    vw.release()
    print(f"wrote {a.out}  ({len(frame_ids)} frames, {len(idxs)} tracks, layout={a.layout})")


if __name__ == "__main__":
    main()
