"""CA-1M side-by-side OFF vs ON temporal-channel visualization.

Picks a val video, runs temporal-OFF and temporal-ON Track C refiners on its
cached features, decodes per-frame 3D boxes to camera frame, projects with the
ViPE step-1 intrinsics, and writes an mp4 with two panels:
  LEFT  : GT (green) + temporal-OFF prediction (red)
  RIGHT : GT (green) + temporal-ON prediction (blue)
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import cv2
import numpy as np
import torch
from vis4d.op.geometry.rotation import quaternion_to_matrix

# CA-1M 3D box edges (vis4d/coco3d canonical 8-corner order)
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def project(corners_cam: np.ndarray, K: np.ndarray):
    """corners_cam [8,3] camera frame → [8,2] image px (NaN if behind cam)."""
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
        # cheap on-screen check
        if max(x1, x2) < -50 or min(x1, x2) > w + 50 or max(y1, y2) < -50 or min(y1, y2) > h + 50:
            continue
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def corners_from_cdr(center, dims, R):
    """center [3], dims [3] full extents, R [3,3] (box-local→cam) → 8 corners [8,3] cam."""
    hx, hy, hz = dims / 2
    local = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[+hx,+hy,-hz],[-hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[+hx,+hy,+hz],[-hx,+hy,+hz],
    ], dtype=np.float32)
    return (R @ local.T).T + center


def infer(refiner, ds, dev='cuda'):
    """Returns list (one per track) of [T, 10] = (center3, dims3, quat4)."""
    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            p = ds[i]
            g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = refiner(
                    hidden_states=g['hidden'].float(), ray_embeddings=g['ray'].float(),
                    depth_latents=g['depth'].float(), pred_box_2d=g['box2d'],
                    intrinsics=g['K'], box_repr=g['box_repr'],
                    timestamps=g['ts'], measured_mask=g['measured'],
                    input_hw=g['input_hw'],
                )
            dec = refiner.decode_layer(o['reg'][-1, :, 0, :].float(), g['box2d'], g['K'], g['input_hw'])
            out.append(dec.float().cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/weka/oe-training-default/jasonr/3d_box/3d_boundingbox_detection/video_3d_box/itw_3dbox_det')
    ap.add_argument('--off_ckpt', required=True)
    ap.add_argument('--on_ckpt', required=True)
    ap.add_argument('--off_no_traj_encoder', type=int, default=1,
                    help='off model uses --no_traj_encoder 1 by default (v11-style)')
    ap.add_argument('--on_multi_token', type=int, default=1,
                    help='on model uses multi-token + kv-norm + warm-start (v15-style)')
    ap.add_argument('--cache_dir', default='outputs/track_c_feature_cache')
    ap.add_argument('--val_split_file', default='outputs/track_c/val_split.json')
    ap.add_argument('--video_id', default='',
                    help='if blank, picks val video with most cached tracks')
    ap.add_argument('--out_dir', default='outputs/track_c/viz_temporal_ca1m')
    ap.add_argument('--fps', type=int, default=10)
    args = ap.parse_args()

    # resolve paths
    if not os.path.isabs(args.cache_dir): args.cache_dir = f'{args.root}/{args.cache_dir}'
    if not os.path.isabs(args.val_split_file): args.val_split_file = f'{args.root}/{args.val_split_file}'
    if not os.path.isabs(args.out_dir): args.out_dir = f'{args.root}/{args.out_dir}'
    os.makedirs(args.out_dir, exist_ok=True)

    # --- pick video ---
    _split = json.load(open(args.val_split_file))
    if isinstance(_split, dict):
        _split = _split.get('val_video_ids', _split.get('val_videos', []))
    val_vids = set(str(v) for v in _split)
    by_vid = {}
    for p in glob.glob(f'{args.cache_dir}/traj/*.pt'):
        vid = os.path.basename(p).split('__')[0]
        if vid in val_vids:
            by_vid.setdefault(vid, []).append(p)
    if args.video_id:
        VID = args.video_id
    else:
        # pick val video with most cached tracks for visual richness
        VID = max(by_vid.keys(), key=lambda v: len(by_vid[v]))
    paths = sorted(by_vid[VID])
    print(f'[viz] CA-1M video {VID}: {len(paths)} cached tracks', flush=True)
    if not paths:
        print(f'NO TRACKS for {VID}'); return

    # --- ts dirs (step1_index → wide image path) ---
    wide_dirs = sorted(glob.glob(f'{args.root}/outputs/ca1m_extracted/videos/{VID}/*.wide'),
                       key=lambda p: int(os.path.basename(p).split('.')[0]))
    # underscore-prefixed dirs are duplicates — drop them
    wide_dirs = [d for d in wide_dirs if not os.path.basename(d).startswith('_')]
    print(f'[viz] {len(wide_dirs)} wide/ dirs found', flush=True)
    if not wide_dirs:
        print('ABORT: no wide/ dirs'); return

    # --- step1 K (per-frame; ViPE estimated for CA-1M) ---
    K_all = np.load(f'{args.root}/outputs/step1/{VID}/intrinsics.npy').astype(np.float32)
    assert K_all.shape[0] == len(wide_dirs), \
        f'K_all.shape[0]={K_all.shape[0]} != n_wide_dirs={len(wide_dirs)}'

    # --- build dataset (preload cache) ---
    sys.path.insert(0, f'{args.root}/WildDet3D')
    from wilddet3d.track_c import TrackCRefiner
    from wilddet3d.track_c.dataset import CachedTrackCDataset
    ds = CachedTrackCDataset(args.cache_dir, paths, preload=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- OFF model ---
    off = TrackCRefiner(
        reg_residual_from_prior=False,
        use_temporal_modules=not bool(args.off_no_traj_encoder),
    ).to(dev)
    off.load_state_dict(torch.load(args.off_ckpt, map_location='cpu', weights_only=False)['refiner'], strict=True)
    off.eval()
    print('[viz] OFF model loaded', flush=True)
    off_preds = infer(off, ds, dev)

    # --- ON model ---
    on = TrackCRefiner(
        reg_residual_from_prior=False,
        use_temporal_modules=True,
        use_temporal_kv_norm=True,
        temporal_multi_token=bool(args.on_multi_token),
    ).to(dev)
    on.load_state_dict(torch.load(args.on_ckpt, map_location='cpu', weights_only=False)['refiner'], strict=True)
    on.eval()
    print('[viz] ON model loaded', flush=True)
    on_preds = infer(on, ds, dev)

    # --- collect per-(frame_idx) records: GT + off + on per track ---
    by_frame = {}  # step1_index -> [(gt_corners, off_corners, on_corners), ...]
    for i, traj in enumerate(ds.traj_cache):
        fr = traj['frame_index'].cpu().numpy()
        gtc = traj['gt_center'].cpu().numpy()
        gtd = traj['gt_dims'].cpu().numpy()
        gtR = quaternion_to_matrix(traj['gt_quat'].float()).cpu().numpy()
        ofc = off_preds[i]; onc = on_preds[i]
        for t, idx in enumerate(fr):
            if gtc[t, 2] <= 1e-3: continue
            off_R = quaternion_to_matrix(torch.from_numpy(ofc[t:t+1, 6:10])).cpu().numpy()[0]
            on_R = quaternion_to_matrix(torch.from_numpy(onc[t:t+1, 6:10])).cpu().numpy()[0]
            gtcorn = corners_from_cdr(gtc[t], gtd[t], gtR[t])
            offcorn = corners_from_cdr(ofc[t, 0:3], ofc[t, 3:6], off_R)
            oncorn = corners_from_cdr(onc[t, 0:3], onc[t, 3:6], on_R)
            by_frame.setdefault(int(idx), []).append((gtcorn, offcorn, oncorn))
    frame_ids = sorted(by_frame.keys())
    if not frame_ids:
        print('ABORT: no valid GT frames in any track'); return

    # --- render side-by-side mp4 ---
    img0 = cv2.imread(f'{wide_dirs[0]}/image.png')
    H, W = img0.shape[:2]
    out_path = f'{args.out_dir}/ca1m_{VID}_sidebyside.mp4'
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                         args.fps, (W * 2, H))
    n_drawn = 0
    for fi in frame_ids:
        if fi >= len(wide_dirs): continue
        img = cv2.imread(f'{wide_dirs[fi]}/image.png')
        if img is None or img.shape[:2] != (H, W): continue
        Ki = K_all[fi]
        imgL = img.copy(); imgR = img.copy()
        for gtcorn, offcorn, oncorn in by_frame[fi]:
            gt2d  = project(gtcorn,  Ki)
            off2d = project(offcorn, Ki)
            on2d  = project(oncorn,  Ki)
            draw_box(imgL, gt2d,  (0, 255, 0), 2)
            draw_box(imgL, off2d, (0, 0, 255), 2)   # red BGR
            draw_box(imgR, gt2d,  (0, 255, 0), 2)
            draw_box(imgR, on2d,  (255, 0, 0), 2)   # blue BGR
        cv2.putText(imgL, 'GT (green) + OFF (red)  v11_no_traj_encoder',
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(imgR, 'GT (green) + ON (blue)  v15_multitoken',
                    (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(cv2.hconcat([imgL, imgR]))
        n_drawn += 1
    vw.release()
    print(f'[viz] wrote {out_path} ({n_drawn}/{len(frame_ids)} frames; image {W}x{H}, panel {W*2}x{H})',
          flush=True)


if __name__ == '__main__':
    main()
