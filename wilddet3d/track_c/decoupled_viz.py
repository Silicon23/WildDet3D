"""Decoupled-pipeline viz: raw Track C vs Track C + Track A post-hoc smoother.

Renders side-by-side on Waymo frames from the prediction dumps (no model inference):
  LEFT  = GT (green) + raw Track C (red)       — accurate-but-jittery per-frame
  RIGHT = GT (green) + Track C + Kalman (blue)  — post-hoc smoothed (the shipped pipeline)

raw dump:      W1_waymo_val_predictions.pt  (list; per-traj pred_/gt_ {center,dims,R}, camera frame)
smoothed dump: W1_kalman_*.pt               (dict[vid/obj]; smoothed_{center,dims,R}, camera frame)

Usage:
  python -m wilddet3d.track_c.decoupled_viz --seg <SEG> --raw <raw.pt> --smoothed <sm.pt> --out <mp4>
"""
import sys, argparse, os, numpy as np, torch, cv2
WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
sys.path.insert(0, WD)
from wilddet3d.track_c.waymo_viz import corners_cam, project, draw_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--smoothed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outputs", default="/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    raw = torch.load(a.raw, weights_only=False)["trajectories"]
    sm = torch.load(a.smoothed, weights_only=False)["trajectories"]  # dict[vid/obj]
    # per-frame: list of (gt10, raw10, sm10) for tracks in this segment
    by_frame = {}
    for t in raw:
        if t["video_id"] != a.seg:
            continue
        key = f"{t['video_id']}/{t['object_id']}"
        s = sm.get(key)
        fidx = np.asarray(t["frame_index"])
        sm_fi = {int(f): i for i, f in enumerate(np.asarray(s["frame_index"]))} if s else {}
        for j, fi in enumerate(fidx):
            fi = int(fi)
            gt = np.concatenate([t["gt_center"][j], t["gt_dims"][j], _q(t["gt_R"][j])])
            rw = np.concatenate([t["pred_center"][j], t["pred_dims"][j], _q(t["pred_R"][j])])
            sj = sm_fi.get(fi)
            sb = (np.concatenate([s["smoothed_center"][sj], s["smoothed_dims"][sj], _q(s["smoothed_R"][sj])])
                  if sj is not None else None)
            by_frame.setdefault(fi, []).append((gt, rw, sb))

    K = np.load(f"{a.outputs}/step1_waymo/{a.seg}/intrinsics.npy")
    per_frame_K = K.ndim == 3
    frames_dir = f"{a.outputs}/step1_waymo/{a.seg}/frames"
    fids = sorted(by_frame.keys())
    im0 = cv2.imread(f"{frames_dir}/{fids[0]:06d}.jpg"); H, W = im0.shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W * 2, H))
    GREEN, RED, BLUE = (0, 200, 0), (0, 0, 230), (230, 80, 0)

    def cc(x):
        from vis4d.op.geometry.rotation import quaternion_to_matrix
        return corners_cam(x[0:3], x[3:6], quaternion_to_matrix(torch.tensor(x[6:10])).numpy())

    for fi in fids:
        img = cv2.imread(f"{frames_dir}/{fi:06d}.jpg")
        if img is None:
            continue
        Ki = K[fi] if per_frame_K else K
        L, Rp = img.copy(), img.copy()
        for gt, rw, sb in by_frame[fi]:
            draw_box(L, project(cc(gt), Ki), GREEN, 2)
            draw_box(L, project(cc(rw), Ki), RED, 2)
            draw_box(Rp, project(cc(gt), Ki), GREEN, 2)
            if sb is not None:
                draw_box(Rp, project(cc(sb), Ki), BLUE, 2)
        cv2.putText(L, "raw Track C (red) vs GT (green)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Rp, "Track C + Kalman (blue) vs GT (green)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(cv2.hconcat([L, Rp]))
    vw.release()
    print(f"wrote {a.out} ({len(fids)} frames)")


def _q(R):
    from vis4d.op.geometry.rotation import matrix_to_quaternion
    return matrix_to_quaternion(torch.tensor(np.asarray(R), dtype=torch.float32)).numpy()


if __name__ == "__main__":
    main()
