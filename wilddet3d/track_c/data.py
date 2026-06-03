"""Track C training-data loader (CA-1M pipeline outputs).

Builds, for one ``(video_id, object_id)`` trajectory, everything Track C needs:

- the **prior trajectory** in absolute camera-frame box representation
  ``box_repr = [center(3), log_dims(3), rot_6d(6)]`` at every *refine frame*
  (densified: measured keyframes carry the real prior box, gaps are interpolated),
  plus ``measured_mask`` and real ``timestamps`` (seconds),
- per refine frame: the **geo-prompt 2D box** (CA-1M ``box_2d_proj`` for CA-1M
  training, per design §7.1; the Step-2 mask bbox is the inference-time prompt),
  the **GT camera-frame 3D box** ``(center, dims, quat)`` (regression target),
  intrinsics, and the image / depth paths.

Coordinate facts (verified): CA-1M ``wide/instances.json`` is camera-frame OpenCV;
``position``=center, ``scale``=FULL extents, ``R``=box->camera. Prior source
``step4`` reads per-frame ``box.{center_cam, R_cam, obb_extents}``; ``step5`` reads
the Kalman ``trajectory.npz`` (WORLD frame) and transforms to camera via Step-1
``extrinsics`` (world->cam). step1 is 1:1 with CA-1M frames (frame_skip=1).
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def _rot6d_from_R(R: np.ndarray) -> np.ndarray:
    """First two rows of R, flattened (matches wilddet3d matrix_to_rotation_6d)."""
    return R[:2, :].reshape(-1).astype(np.float32)


def _quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w,x,y,z). Matches vis4d matrix_to_quaternion."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


@dataclass
class FrameSample:
    step1_index: int
    ts_ns: int
    measured: bool                 # has a prior (Step-4/5) measurement
    box_repr: np.ndarray           # [12] absolute cam-frame prior box (filled)
    gt_center: np.ndarray          # [3]
    gt_dims: np.ndarray            # [3] full extents
    gt_quat_wxyz: np.ndarray       # [4]
    prompt_box_xyxy: List[float]   # original-image px geo prompt
    intrinsics: np.ndarray         # [3,3] original-image space
    image_path: str
    depth_path: Optional[str]


@dataclass
class TrajectorySample:
    video_id: str
    object_id: str
    category: str
    frames: List[FrameSample] = field(default_factory=list)

    def box_repr(self) -> np.ndarray:
        return np.stack([f.box_repr for f in self.frames], 0)            # [T,12]

    def timestamps_sec(self) -> np.ndarray:
        t = np.array([f.ts_ns for f in self.frames], np.float64)
        return ((t - t.min()) / 1e9).astype(np.float32)                  # [T]

    def measured_mask(self) -> np.ndarray:
        return np.array([f.measured for f in self.frames], bool)         # [T]


def _camera_box_repr(center: np.ndarray, dims: np.ndarray, R: np.ndarray) -> np.ndarray:
    """[center(3), log_dims(3), rot6d(6)] -> [12]. dims = full extents (>0)."""
    log_dims = np.log(np.clip(dims, 1e-4, None)).astype(np.float32)
    return np.concatenate([center.astype(np.float32), log_dims, _rot6d_from_R(R)])


def _interp_box_repr(timeline_idx: np.ndarray, meas_idx: np.ndarray,
                     meas_repr: np.ndarray) -> np.ndarray:
    """Fill box_repr for every timeline frame.

    center(0:3) + log_dims(3:6): linear interp over frame index; rot6d(6:12):
    nearest measured (avoids invalid linear blends of rotation rows). Endpoints
    clamp to the nearest measurement.
    """
    T = len(timeline_idx)
    out = np.zeros((T, 12), np.float32)
    # linear interp for center + log_dims
    for d in range(6):
        out[:, d] = np.interp(timeline_idx, meas_idx, meas_repr[:, d])
    # nearest measured for rot6d
    nn = np.searchsorted(meas_idx, timeline_idx)
    nn = np.clip(nn, 0, len(meas_idx) - 1)
    for ti in range(T):
        c = nn[ti]
        if c > 0 and abs(meas_idx[c - 1] - timeline_idx[ti]) < abs(meas_idx[c] - timeline_idx[ti]):
            c = c - 1
        out[ti, 6:] = meas_repr[c, 6:]
    return out


def _load_clip_range(outputs_dir: str, video_id: str) -> Optional[Dict]:
    path = f"{outputs_dir}/step2_5_clip_select/clip_index.jsonl"
    if not os.path.exists(path):
        return None
    for line in open(path):
        j = json.loads(line)
        if str(j.get("video_id")) == str(video_id):
            return j
    return None


def _step4_camera_boxes(outputs_dir: str, video_id: str, object_id: str) -> Dict[int, np.ndarray]:
    """step1_index -> absolute camera-frame box_repr [12] from Step-4."""
    meta = json.load(open(f"{outputs_dir}/step4/{video_id}/{object_id}/meta.json"))
    out = {}
    for fr in meta["frames"]:
        if fr.get("status") != "ok":
            continue
        box = fr.get("box")
        if not isinstance(box, dict) or "center_cam" not in box:
            continue
        center = np.array(box["center_cam"], np.float32)
        R = np.array(box["R_cam"], np.float32)
        dims = np.array(box.get("obb_extents", box.get("size")), np.float32)
        out[int(fr["step1_index"])] = _camera_box_repr(center, dims, R)
    return out


def load_trajectory(
    outputs_dir: str,
    video_id: str,
    object_id: str,
    ca1m_videos_dir: Optional[str] = None,
    step1_dir: Optional[str] = None,
    prior_source: str = "step4",
    max_frames: Optional[int] = None,
) -> Optional[TrajectorySample]:
    """Load one (video, object) trajectory. Returns None if unusable."""
    ca1m_videos_dir = ca1m_videos_dir or f"{outputs_dir}/ca1m_extracted/videos"
    step1_dir = step1_dir or f"{outputs_dir}/step1/{video_id}"

    clip = _load_clip_range(outputs_dir, video_id)
    if clip is None:
        return None
    start, end = int(clip["start_frame_idx"]), int(clip["end_frame_idx"])

    wdirs = sorted(
        glob.glob(f"{ca1m_videos_dir}/{video_id}/*.wide"),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    if not wdirs:
        return None
    ts_ns = [int(os.path.basename(p).split(".")[0]) for p in wdirs]
    n_frames = len(wdirs)

    K_all = np.load(f"{step1_dir}/intrinsics.npy")          # [N,3,3]

    if prior_source == "step4":
        prior = _step4_camera_boxes(outputs_dir, video_id, object_id)
    else:
        raise NotImplementedError(f"prior_source={prior_source} (step5 TODO)")
    if not prior:
        return None

    # Refine frames: in clip range, GT-present, prior coverage available.
    end = min(end, n_frames)
    refine = []
    category = ""
    for idx in range(start, end):
        insts = json.load(open(f"{wdirs[idx]}/instances.json"))
        obj = next((o for o in insts if o["id"] == object_id), None)
        if obj is None:
            continue
        category = obj.get("category", category)
        refine.append((idx, obj))
    if len(refine) < 3:
        return None
    if max_frames is not None and len(refine) > max_frames:
        refine = refine[:max_frames]

    timeline_idx = np.array([i for i, _ in refine], np.float64)
    meas_idx = np.array(sorted(prior.keys()), np.float64)
    meas_repr = np.stack([prior[int(i)] for i in meas_idx], 0)
    filled = _interp_box_repr(timeline_idx, meas_idx, meas_repr)
    meas_set = set(int(i) for i in meas_idx)

    sample = TrajectorySample(video_id=video_id, object_id=object_id, category=category)
    for k, (idx, obj) in enumerate(refine):
        center = np.array(obj["position"], np.float32)
        dims = np.array(obj["scale"], np.float32)
        R = np.array(obj["R"], np.float32)
        # measured frame -> use the actual prior box, else the interpolated fill
        box_repr = prior[idx] if idx in meas_set else filled[k]
        depth_p = f"{step1_dir}/depth/{idx:06d}.npy"
        sample.frames.append(FrameSample(
            step1_index=idx,
            ts_ns=ts_ns[idx],
            measured=idx in meas_set,
            box_repr=box_repr.astype(np.float32),
            gt_center=center,
            gt_dims=dims,
            gt_quat_wxyz=_quat_wxyz_from_R(R),
            prompt_box_xyxy=[float(x) for x in obj["box_2d_proj"]],
            intrinsics=K_all[idx].astype(np.float32),
            image_path=f"{wdirs[idx]}/image.png",
            depth_path=depth_p if os.path.exists(depth_p) else None,
        ))
    return sample


def list_trajectories(outputs_dir: str, step_dir: str = "step4") -> List[tuple]:
    """All (video_id, object_id) under outputs/<step_dir>/."""
    base = f"{outputs_dir}/{step_dir}"
    out = []
    for vid in sorted(os.listdir(base)):
        vp = f"{base}/{vid}"
        if not os.path.isdir(vp):
            continue
        for obj in sorted(os.listdir(vp)):
            if os.path.isdir(f"{vp}/{obj}"):
                out.append((vid, obj))
    return out
