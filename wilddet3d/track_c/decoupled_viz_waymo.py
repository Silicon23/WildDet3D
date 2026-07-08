"""Decoupled-pipeline Waymo viz: raw Track C vs Track C + Track A Kalman.

Supports single-panel (default) or side-by-side layouts. For WLK1 (GT-K + GT-LiDAR),
projects with GT intrinsics from waymo_production/<seg>/gt/intrinsics.npy (since
the predictions live in the GT-K frame the model was trained in).

Frames read from outputs/step1_waymo/<SEG>/frames/<idx:06d>.jpg.

Schemas:
  raw:      list of traj-dicts (video_id, object_id, frame_index, pred_/gt_ {center,dims,R})
  smoothed: dict keyed by "<vid>/<obj>" with {frame_index, smoothed_{center,dims,R}}
"""
from __future__ import annotations
import argparse, os, cv2, numpy as np, torch


EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def corners_from_cdr(center, dims, R):
    hx, hy, hz = dims / 2
    local = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[+hx,+hy,-hz],[-hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[+hx,+hy,+hz],[-hx,+hy,+hz],
    ], dtype=np.float32)
    return (R @ local.T).T + center


def project(corners_cam, K):
    out = np.full((8, 2), np.nan)
    for i, (x, y, z) in enumerate(corners_cam):
        if z > 0.1:
            out[i] = [K[0,0] * x / z + K[0,2], K[1,1] * y / z + K[1,2]]
    return out


def draw_box(img, pts2d, color, thickness=2):
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
    ap.add_argument("--raw", required=True, help="e.g. WLK1_waymo_val_predictions.pt")
    ap.add_argument("--smoothed", default="", help="e.g. WLK1_kalman_rotdims_const.pt "
                    "(required for --source smoothed and for side-by-side layout)")
    ap.add_argument("--seg", default="", help="segment id; empty = pick busiest val seg")
    ap.add_argument("--intrinsics_source", default="gt", choices=("gt", "vipe"),
                    help="gt: waymo_production/<seg>/gt/intrinsics.npy (WLK1); vipe: step1_waymo (W1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--single_panel", type=int, default=1)
    ap.add_argument("--source", default="smoothed",
                    choices=("input", "pred", "smoothed"),
                    help="single-panel color source: input=Step-4 FoundationPose prior "
                         "(before WildDet3D); pred=Track C output (post-WildDet3D); "
                         "smoothed=Track C + Track A Kalman (default; shipped result)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    raw = torch.load(a.raw, weights_only=False)["trajectories"]  # list
    need_smoothed = a.source == "smoothed" or not a.single_panel
    if need_smoothed and not a.smoothed:
        raise SystemExit("--smoothed required for --source smoothed or side-by-side layout")
    sm = torch.load(a.smoothed, weights_only=False)["trajectories"] if a.smoothed else {}  # dict[vid/obj]

    # pick busiest val seg if not specified
    if not a.seg:
        import collections
        counts = collections.Counter(t["video_id"] for t in raw)
        a.seg = counts.most_common(1)[0][0]
    tracks = [t for t in raw if t["video_id"] == a.seg]
    print(f"[viz] Waymo seg {a.seg}: {len(tracks)} tracks", flush=True)

    # intrinsics
    if a.intrinsics_source == "gt":
        K_all = np.load(f"{a.outputs}/waymo_production/{a.seg}/gt/intrinsics.npy").astype(np.float32)
    else:
        K_all = np.load(f"{a.outputs}/step1_waymo/{a.seg}/intrinsics.npy").astype(np.float32)
    per_frame_K = K_all.ndim == 3
    frames_dir = f"{a.outputs}/step1_waymo/{a.seg}/frames"

    # per-frame accumulation. inp_ = Step-4 FoundationPose (pre-WildDet3D) prior.
    by_frame = {}
    for t in tracks:
        key = f"{t['video_id']}/{t['object_id']}"
        s = sm.get(key)
        fidx = np.asarray(t["frame_index"])
        rc = np.asarray(t["pred_center"]); rd = np.asarray(t["pred_dims"]); rR = np.asarray(t["pred_R"])
        ic = np.asarray(t["input_center"]); id_ = np.asarray(t["input_dims"]); iR = np.asarray(t["input_R"])
        gc = np.asarray(t["gt_center"]);   gd = np.asarray(t["gt_dims"]);   gR = np.asarray(t["gt_R"])
        if s is not None:
            sfi = np.asarray(s["frame_index"])
            sc = np.asarray(s["smoothed_center"]); sd = np.asarray(s["smoothed_dims"]); sR = np.asarray(s["smoothed_R"])
            s_idx = {int(f): i for i, f in enumerate(sfi)}
        else:
            s_idx = {}
        for j, fi in enumerate(fidx):
            fi = int(fi)
            if gc[j, 2] <= 1e-3:
                continue
            # skip frames where the Step-4 prior itself is invalid (interpolated
            # sentinel = center exactly at origin, or FP-failed with z <= 0)
            inp_valid = bool(np.linalg.norm(ic[j]) > 1e-3 and ic[j, 2] > 0.1)
            gtc = corners_from_cdr(gc[j], gd[j], gR[j])
            rwc = corners_from_cdr(rc[j], rd[j], rR[j])
            inpc = corners_from_cdr(ic[j], id_[j], iR[j]) if inp_valid else None
            si = s_idx.get(fi)
            smc = corners_from_cdr(sc[si], sd[si], sR[si]) if si is not None else rwc
            by_frame.setdefault(fi, []).append((gtc, rwc, smc, inpc))
    fids = sorted(by_frame.keys())
    if not fids:
        print("ABORT: no valid frames"); return

    im0 = cv2.imread(f"{frames_dir}/{fids[0]:06d}.jpg")
    if im0 is None:
        print(f"ABORT: cannot read {frames_dir}/{fids[0]:06d}.jpg"); return
    H, W = im0.shape[:2]
    panel_w = W if a.single_panel else W * 2
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (panel_w, H))
    GREEN, RED, BLUE = (0, 255, 0), (0, 0, 255), (255, 0, 0)
    n = 0
    for fi in fids:
        img = cv2.imread(f"{frames_dir}/{fi:06d}.jpg")
        if img is None:
            continue
        Ki = K_all[fi] if per_frame_K else K_all
        if a.single_panel:
            R = img.copy()
            for gt_c, raw_c, sm_c, inp_c in by_frame[fi]:
                draw_box(R, project(gt_c, Ki), GREEN, 2)
                if a.source == "input":
                    if inp_c is not None:
                        draw_box(R, project(inp_c, Ki), BLUE, 2)
                elif a.source == "pred":
                    draw_box(R, project(raw_c, Ki), BLUE, 2)
                else:  # smoothed
                    draw_box(R, project(sm_c, Ki), BLUE, 2)
            label = {
                "input": "Step-4 FoundationPose prior (blue, before WildDet3D)  vs  GT (green)",
                "pred": "Track C (blue, post-WildDet3D)  vs  GT (green)",
                "smoothed": "Track C + Track A Kalman (blue)  vs  GT (green)",
            }[a.source]
            cv2.putText(R, label,
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(R)
        else:
            L = img.copy(); Rp = img.copy()
            for gt_c, raw_c, sm_c, inp_c in by_frame[fi]:
                draw_box(L, project(gt_c, Ki), GREEN, 2); draw_box(L, project(raw_c, Ki), RED, 2)
                draw_box(Rp, project(gt_c, Ki), GREEN, 2); draw_box(Rp, project(sm_c, Ki), BLUE, 2)
            cv2.putText(L, "raw Track C (red) vs GT (green)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(Rp, "Track C + Track A Kalman (blue) vs GT (green)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            vw.write(cv2.hconcat([L, Rp]))
        n += 1
    vw.release()
    print(f"[viz] wrote {a.out} ({n} frames)", flush=True)


if __name__ == "__main__":
    main()
