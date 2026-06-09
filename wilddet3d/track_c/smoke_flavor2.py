"""Smoke-test the Flavor-2 multi-token temporal path (forward + backward)."""
import sys, torch
WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
OUTS = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs"
sys.path.insert(0, WD)
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import (CachedTrackCDataset, collate_trajs,
                                       list_cached_trajectories, split_by_video)

dev = "cuda"
CKPT = f"{WD}/ckpt/wilddet3d_alldata_all_prompt_v1.0.pt"
paths = list_cached_trajectories(f"{OUTS}/track_c_feature_cache")
_, val_paths, _ = split_by_video(paths, val_frac=0.08)
ds = CachedTrackCDataset(f"{OUTS}/track_c_feature_cache", val_paths[:2], preload=False)
packs = [ds[i] for i in range(2)]
batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_trajs(packs).items()}
print(f"batch: K={len(packs)} sizes={[p['box_repr'].shape[0] for p in packs]} "
      f"sum_T={int((~batch['pad_mask']).sum())}")

r = TrackCRefiner(reg_residual_from_prior=False, use_temporal_modules=True,
                  use_temporal_kv_norm=True, temporal_multi_token=True).to(dev)
info = r.load_pretrained_head(CKPT)
r.warm_start_temporal_from_depth()
r.finalize_init()
print(f"head load new(missing)={len(info['missing'])}; multi_token={r.head.temporal_multi_token}")

# ---- identity-at-init: gate=0 => multi-token temporal contributes nothing ----
r.eval()
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_mt = r.forward_vectorized(batch)["reg"].float()
    # temporal-off reference: same refiner, temporal modules bypassed
    r.head.use_temporal_prompt = False
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_off = r.forward_vectorized(batch)["reg"].float()
    r.head.use_temporal_prompt = True
id_err = (out_mt - out_off).abs().max().item()
print(f"[init] max|reg(multi-token) - reg(temporal-off)| = {id_err:.3e} (expect ~0 at gate=0)")
print(f"[init] reg finite: {torch.isfinite(out_mt).all().item()} (no NaN from masked attn)")

# ---- backward: grads reach the temporal/cross-attn + PE-consuming params ----
r.train()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = r.forward_vectorized(batch)
loss = out["reg"].float().pow(2).mean()
loss.backward()
def gnorm(sub):
    g = [p.grad.norm().item() for n, p in r.named_parameters()
         if sub in n and p.grad is not None and p.grad.abs().sum() > 0]
    return (len(g), round(sum(g), 5))
print(f"[grad] prompt_temporal: {gnorm('prompt_temporal')}")
print(f"[grad] project_temporal: {gnorm('project_temporal')}")
print(f"[grad] temporal_kv_norm: {gnorm('temporal_kv_norm')}")
print(f"[grad] temporal_gate: {gnorm('temporal_gate')}")
print(f"[grad] traj_encoder: {gnorm('traj_encoder')}")
print("SMOKE OK")

# ---- with gate>0, grads must reach the whole temporal branch (path connected) ----
r.zero_grad(set_to_none=True)
with torch.no_grad():
    for g in r.head.temporal_gate:
        g.fill_(0.1)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out2 = r.forward_vectorized(batch)
out2["reg"].float().pow(2).mean().backward()
print("[gate=0.1] prompt_temporal:", gnorm("prompt_temporal"))
print("[gate=0.1] project_temporal:", gnorm("project_temporal"))
print("[gate=0.1] temporal_kv_norm:", gnorm("temporal_kv_norm"))
print("[gate=0.1] traj_encoder:", gnorm("traj_encoder"))
print("CONNECTIVITY OK")
