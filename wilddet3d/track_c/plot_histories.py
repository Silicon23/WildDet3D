"""Render Track C loss-curve images from per-run history.csv files.

Per run: train_loss + held-out iou3d vs epoch (twin axis) -> runs/<id>/loss_curve.png
Combined: all runs' iou3d vs epoch, and train_loss vs epoch -> plots/*.png
"""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs/track_c/runs"
PLOTS = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs/track_c/plots"
os.makedirs(PLOTS, exist_ok=True)
order = ["v1", "v2_from_pretrained", "v3_no_temporal", "v4_deriv_on",
         "v5_deriv_off", "v6_deriv_gentle", "v7_deriv_moderate"]


def load(run):
    p = f"{RUNS}/{run}/history.csv"
    if not os.path.exists(p):
        return None
    ep, tl, iou = [], [], []
    for r in csv.DictReader(open(p)):
        ep.append(int(r["epoch"])); tl.append(float(r["train_loss"]))
        iou.append(float(r["eval_iou3d"]) if r["eval_iou3d"] else None)
    return ep, tl, iou


data = {r: load(r) for r in order if load(r)}

# per-run twin-axis plot
for run, (ep, tl, iou) in data.items():
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ep, tl, "b-o", ms=3, label="train_loss"); ax1.set_xlabel("epoch")
    ax1.set_ylabel("train_loss", color="b"); ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ee = [(e, v) for e, v in zip(ep, iou) if v is not None]
    if ee:
        ax2.plot([e for e, _ in ee], [v for _, v in ee], "r-s", ms=4, label="held-out iou3d")
    ax2.set_ylabel("held-out iou3d", color="r"); ax2.tick_params(axis="y", labelcolor="r")
    plt.title(run); fig.tight_layout(); fig.savefig(f"{RUNS}/{run}/loss_curve.png", dpi=110); plt.close(fig)

# combined iou3d
fig, ax = plt.subplots(figsize=(7, 4.5))
for run, (ep, tl, iou) in data.items():
    ee = [(e, v) for e, v in zip(ep, iou) if v is not None]
    if ee:
        ax.plot([e for e, _ in ee], [v for _, v in ee], "-o", ms=4, label=run)
ax.axhline(0.096, ls="--", c="gray", label="baseline (Step-4) 0.096")
ax.set_xlabel("epoch"); ax.set_ylabel("held-out 3D IoU"); ax.set_title("Track C — held-out iou3d")
ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(f"{PLOTS}/eval_iou3d_all.png", dpi=120); plt.close(fig)

# combined train_loss
fig, ax = plt.subplots(figsize=(7, 4.5))
for run, (ep, tl, iou) in data.items():
    ax.plot(ep, tl, "-", label=run)
ax.set_xlabel("epoch"); ax.set_ylabel("train_loss"); ax.set_title("Track C — train_loss (not comparable across loss configs)")
ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(f"{PLOTS}/train_loss_all.png", dpi=120); plt.close(fig)
print("wrote per-run loss_curve.png +", PLOTS + "/{eval_iou3d_all,train_loss_all}.png")
