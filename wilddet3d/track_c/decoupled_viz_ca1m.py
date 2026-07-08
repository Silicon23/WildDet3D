"""Decoupled-pipeline CA-1M viz: raw Track C vs Track C + Track A post-hoc smoother.

LEFT panel  : GT (green) + raw Track C (red)         — accurate-but-jittery per-frame
RIGHT panel : GT (green) + Track C + Kalman (blue)   — post-hoc smoothed, shipped pipeline

Reads the shared-686-traj dumps that Track A used to compute the report numbers:
  raw:      outputs/track_c/v11_val_predictions_686.pt        {trajectories: dict[vid/obj]}
  smoothed: outputs/track_c/v11_kalman_smoothed_686.pt        (same schema, smoothed_*)

CA-1M frames come from outputs/ca1m_extracted/videos/<VID>/<TS>.wide/image.png at
native 10 fps (ARKit). Intrinsics from outputs/step1/<VID>/intrinsics.npy (ViPE,
empirically stable on CA-1M indoor).
"""
from __future__ import annotations
import argparse, glob, os, cv2, numpy as np, torch


EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def corners_from_cdr(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """center [3], dims [3] full extents, R [3,3] box-local→cam -> 8 corners [8,3] cam."""
    hx, hy, hz = dims / 2
    local = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[+hx,+hy,-hz],[-hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[+hx,+hy,+hz],[-hx,+hy,+hz],
    ], dtype=np.float32)
    return (R @ local.T).T + center


def project(corners_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """corners [8,3] cam -> [8,2] px (NaN behind camera)."""
    out = np.full((8, 2), np.nan)
    for i, (x, y, z) in enumerate(corners_cam):
        if z > 0.1:
            out[i] = [K[0,0] * x / z + K[0,2], K[1,1] * y / z + K[1,2]]
    return out


def draw_box(img: np.ndarray, pts2d: np.ndarray, color: tuple, thickness: int = 2) -> None:
    h, w = img.shape[:2]
    for a, b in EDGES:
        pa, pb = pts2d[a], pts2d[b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        x1, y1 = int(pa[0]), int(pa[1])
        x2, y2 = int(pb[0]), int(pb[1])
        if max(x1, x2) < -50 or min(x1, x2) > w + 50 or max(y1, y2) < -50 or min(y1, y2) > h + 50:
            continue
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="/weka/oe-training-default/jasonr/3d_box/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs")
    ap.add_argument("--raw", default="")
    ap.add_argument("--smoothed", default="")
    ap.add_argument("--video_id", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--single_panel", type=int, default=0,
                    help="if 1, draw only GT + smoothed on a single-width panel (no raw)")
    a = ap.parse_args()
    if not a.raw: a.raw = f"{a.outputs}/track_c/v11_val_predictions_686.pt"
    if not a.smoothed: a.smoothed = f"{a.outputs}/track_c/v11_kalman_smoothed_686.pt"

    raw = torch.load(a.raw, weights_only=False)["trajectories"]
    sm = torch.load(a.smoothed, weights_only=False)["trajectories"]
    # normalize schemas: smoothed dict is keyed by "vid/obj"; raw dict is too
    # pick video with most tracks in intersection
    import collections
    common = set(raw.keys()) & set(sm.keys())
    counts = collections.Counter(k.split("/")[0] for k in common)
    if a.video_id:
        VID = a.video_id
    else:
        VID = counts.most_common(1)[0][0]
    if not a.out:
        a.out = f"{a.outputs}/track_c/viz_decoupled/ca1m_{VID}_raw_vs_kalman.mp4"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    tracks = [k for k in common if k.startswith(f"{VID}/")]
    print(f"[viz] CA-1M video {VID}: {len(tracks)} tracks (paired-686 subset)", flush=True)
    if not tracks:
        print("ABORT: no tracks"); return

    # per-frame intrinsics (step1 ViPE — stable on CA-1M indoor)
    K_all = np.load(f"{a.outputs}/step1/{VID}/intrinsics.npy").astype(np.float32)
    # ordered .wide dirs (drop underscore-prefixed duplicates)
    wide_dirs = sorted(
        glob.glob(f"{a.outputs}/ca1m_extracted/videos/{VID}/*.wide"),
        key=lambda p: int(os.path.basename(p).split(".")[0]))
    wide_dirs = [d for d in wide_dirs if not os.path.basename(d).startswith("_")]
    print(f"[viz] {len(wide_dirs)} wide/ frames, K_all shape {K_all.shape}", flush=True)
    if K_all.shape[0] != len(wide_dirs):
        print(f"⚠ K length {K_all.shape[0]} != n_frames {len(wide_dirs)} — using min", flush=True)

    # collect per-frame records: for each track, decode all frames into (gt, raw, sm) corners
    from vis4d.op.geometry.rotation import quaternion_to_matrix
    by_frame = {}   # step1_index -> list of (gt_corners, raw_corners, sm_corners)
    for k in tracks:
        r, s = raw[k], sm[k]
        fr = r["frame_index"].numpy() if torch.is_tensor(r["frame_index"]) else np.asarray(r["frame_index"])
        rc = np.asarray(r["pred_center"]); rd = np.asarray(r["pred_dims"]); rR = np.asarray(r["pred_R"])
        gc = np.asarray(r["gt_center"]);   gd = np.asarray(r["gt_dims"]);   gR = np.asarray(r["gt_R"])
        sc = np.asarray(s["smoothed_center"]); sd = np.asarray(s["smoothed_dims"]); sR = np.asarray(s["smoothed_R"])
        # frame_index in smoothed dump may differ if a Kalman rejected any frames; align by intersection
        sfr = s["frame_index"].numpy() if torch.is_tensor(s["frame_index"]) else np.asarray(s["frame_index"])
        # index smoothed by frame_index
        s_idx = {int(f): i for i, f in enumerate(sfr)}
        for t, idx in enumerate(fr):
            idx = int(idx)
            if gc[t, 2] <= 1e-3:
                continue
            gtc = corners_from_cdr(gc[t], gd[t], gR[t])
            rwc = corners_from_cdr(rc[t], rd[t], rR[t])
            if idx not in s_idx:
                # smoothed frame missing (unlikely); use raw for the right panel too
                smc = rwc
            else:
                si = s_idx[idx]
                smc = corners_from_cdr(sc[si], sd[si], sR[si])
            by_frame.setdefault(idx, []).append((gtc, rwc, smc))
    frames_sorted = sorted(by_frame.keys())
    if not frames_sorted:
        print("ABORT: no valid frames"); return

    # sample one frame to get image dims
    img0 = cv2.imread(f"{wide_dirs[0]}/image.png")
    if img0 is None:
        print(f"ABORT: cannot read {wide_dirs[0]}/image.png"); return
    H, W = img0.shape[:2]
    panel_w = W if a.single_panel else W * 2
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (panel_w, H))
    n = 0
    for fi in frames_sorted:
        if fi >= len(wide_dirs):
            continue
        img = cv2.imread(f"{wide_dirs[fi]}/image.png")
        if img is None or img.shape[:2] != (H, W):
            continue
        Ki = K_all[min(fi, K_all.shape[0] - 1)]
        if a.single_panel:
            R = img.copy()
            for gt_c, raw_c, sm_c in by_frame[fi]:
                gp = project(gt_c, Ki)
                sp = project(sm_c, Ki)
                draw_box(R, gp, (0, 255, 0), 2)      # GT green
                draw_box(R, sp, (255, 0, 0), 2)      # smoothed blue
            cv2.putText(R, "Track C + Track A Kalman (blue)  vs  GT (green)",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(R)
        else:
            L = img.copy(); R = img.copy()
            for gt_c, raw_c, sm_c in by_frame[fi]:
                gp = project(gt_c, Ki)
                rp = project(raw_c, Ki)
                sp = project(sm_c, Ki)
                draw_box(L, gp, (0, 255, 0), 2)
                draw_box(L, rp, (0, 0, 255), 2)
                draw_box(R, gp, (0, 255, 0), 2)
                draw_box(R, sp, (255, 0, 0), 2)
            cv2.putText(L, "raw Track C (red)",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(R, "+ Track A Kalman (blue)",
                        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(cv2.hconcat([L, R]))
        n += 1
    vw.release()
    print(f"[viz] wrote {a.out} ({n} frames)", flush=True)


if __name__ == "__main__":
    main()
