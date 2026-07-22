"""Regression test for the frames.pt merge bug (2026-07-22).

Failure being guarded against: an incremental precompute pass (e.g. adding a
new category's tracks to a video that already has other-category tracks
cached) rewrites ``frames/<seg>.pt`` with only its own step1 indices, silently
dropping frames referenced by previously-cached traj files. Downstream reads
then raise ``KeyError`` on missing indices. This bit us on 2026-07-22 when the
Waymo pedestrian precompute broke 39 of 239 veh+cyc trajectories across 29
videos.

The fix (see wilddet3d/track_c/precompute.py:_load_or_new_frame_cache): read
the existing frames.pt on precompute entry so writes union rather than
overwrite. This test verifies that helper directly and asserts the union
property end-to-end without needing a GPU or real dataset.

Run: ``pytest tests/test_frame_cache_merge.py -v`` (or plain ``python -m
tests.test_frame_cache_merge`` for the __main__ shim below).
"""

from __future__ import annotations

import os
import tempfile

import torch

from wilddet3d.track_c.precompute import _atomic_save, _load_or_new_frame_cache


def _fake_frame_record(seed: int) -> dict:
    """Small stand-in for the (depth_latents, ray, K, input_hw) dict a real
    precompute writes. Content doesn't matter — only identity does — so use
    tiny tensors so the test file stays small."""
    g = torch.Generator().manual_seed(seed)
    return {
        "depth_latents": torch.randn(4, 8, generator=g, dtype=torch.bfloat16),
        "ray": torch.randn(4, 8, generator=g, dtype=torch.bfloat16),
        "K": torch.eye(3),
        "input_hw": [1008, 1008],
        "_tag": f"seed={seed}",   # unique marker so we can verify identity
    }


def test_load_or_new_no_file_returns_empty(tmp_path):
    """If frames.pt doesn't exist yet, helper returns an empty dict — same
    behaviour as the original ``frame_cache = {}`` initialiser."""
    p = tmp_path / "nonexistent" / "frames" / "video42.pt"
    assert not p.exists()
    assert _load_or_new_frame_cache(str(p)) == {}


def test_load_or_new_returns_existing(tmp_path):
    """If frames.pt exists, its contents are read back."""
    p = tmp_path / "frames" / "video42.pt"
    original = {0: _fake_frame_record(0), 5: _fake_frame_record(5)}
    _atomic_save(original, str(p))
    got = _load_or_new_frame_cache(str(p))
    assert set(got.keys()) == {0, 5}
    assert got[0]["_tag"] == "seed=0"
    assert got[5]["_tag"] == "seed=5"


def test_incremental_pass_preserves_prior_frames(tmp_path):
    """The regression scenario itself: pass A caches step1 indices {0, 2, 4},
    then pass B (a different category on the same seg) needs {2, 6, 8}. The
    resulting frames.pt must contain the *union* {0, 2, 4, 6, 8}. Without the
    merge fix pass B would overwrite and leave only {2, 6, 8} — which is what
    caused the KeyError on frames 0 and 4 for the earlier pass's trajs."""
    p = tmp_path / "frames" / "seg_a.pt"

    # --- pass A: veh+cyc precompute writes frames 0, 2, 4 ---
    pass_a = {i: _fake_frame_record(i) for i in [0, 2, 4]}
    _atomic_save(pass_a, str(p))

    # --- pass B: ped precompute (or any second run) starts, initialises with
    # the merge helper, adds new frames, writes back. This is exactly what the
    # patched frame_cache init in {precompute,waymo_precompute,adt_precompute}
    # now does. ---
    frame_cache = _load_or_new_frame_cache(str(p))
    for i in [2, 6, 8]:
        frame_cache[i] = _fake_frame_record(i + 100)   # different content on the shared idx=2
    _atomic_save(frame_cache, str(p))

    # --- verify: union of indices preserved ---
    final = torch.load(str(p), weights_only=False)
    assert set(final.keys()) == {0, 2, 4, 6, 8}, (
        f"pre-fix bug: keys={sorted(final.keys())}; union expected {{0,2,4,6,8}}")

    # existing-only indices unchanged
    assert final[0]["_tag"] == "seed=0"
    assert final[4]["_tag"] == "seed=4"

    # shared index (2) picks up the newer content — the second pass has the
    # more recent K/depth for that frame. Overwrite-on-collision is correct
    # (avoids stale entries when e.g. we re-run with corrected intrinsics).
    assert final[2]["_tag"] == "seed=102"

    # new indices present
    assert final[6]["_tag"] == "seed=106"
    assert final[8]["_tag"] == "seed=108"


def test_corrupt_file_falls_back_to_empty(tmp_path):
    """If frames.pt is a truncated/garbage file, the helper must not crash —
    just start fresh. Otherwise a bad file blocks all future precomputes for
    that seg."""
    p = tmp_path / "frames" / "bad.pt"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not a valid torch pickle")
    assert _load_or_new_frame_cache(str(p)) == {}


if __name__ == "__main__":
    # Standalone runner so we can `python -m tests.test_frame_cache_merge`
    # even without pytest installed.
    import inspect, pathlib, sys
    n_ok = n_fail = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        with tempfile.TemporaryDirectory() as td:
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = pathlib.Path(td)
            try:
                fn(**kwargs)
                print(f"  OK   {name}")
                n_ok += 1
            except Exception as e:
                print(f"  FAIL {name}: {type(e).__name__}: {e}")
                n_fail += 1
    print(f"{n_ok} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)
