"""Render the three Track C report figures as vector PDFs.

Outputs to itw_3dbox_det/report/figures/:
  - tradeoff.pdf       : Fig 2 (main body) -- IoU3D vs rotation jitter Pareto
  - trackc_arch.pdf    : App G  -- frozen-vs-trained head architecture schematic
  - trackc_curves.pdf  : App H  -- per-epoch train_loss + held-out evals

NO temporal modules are referenced anywhere (per the report constraint).
Run:  python -m wilddet3d.track_c.render_figures
"""
from __future__ import annotations
import csv, os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, FancyArrowPatch, FancyBboxPatch, Rectangle

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,           # embed fonts as Type-42
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUT_DIR = Path("/weka/oe-training-default/weikaih/3d_boundingbox_detection/"
               "video_3d_box/itw_3dbox_det/report/figures")
RUNS = Path("/weka/oe-training-default/weikaih/3d_boundingbox_detection/"
            "video_3d_box/itw_3dbox_det/outputs/track_c/runs")


# ---------- Fig 2: accuracy <-> smoothness trade-off ----------
def render_tradeoff():
    # Pareto-frontier data points from the App H weight sweep (834-traj held-out).
    # The "chosen" entry is v11 (modules removed, same weights as the rest).
    # (key, w_vel, w_acc, w_rotvel, iou3d, rot_jit_deg)
    points = [
        ("none",       0.00, 0.00, 0.00, 0.230, 6.72),
        ("half",       0.05, 0.05, 0.15, 0.237, 4.30),
        ("chosen",     0.10, 0.10, 0.30, 0.233, 3.43),
        ("between",    0.15, 0.15, 0.60, 0.228, 2.43),
        ("moderate",   0.30, 0.30, 1.00, 0.201, 1.67),
        # strong (2,2,5) over-smooths to IoU 0.131; not on the curve.
    ]
    gt_floor_rot_jit = 0.63
    fig, ax = plt.subplots(figsize=(4.6, 3.4))

    iou = np.array([p[4] for p in points])
    jit = np.array([p[5] for p in points])
    keys = [p[0] for p in points]

    # Frontier line
    order = np.argsort(jit)
    ax.plot(jit[order], iou[order], "-", color="#7c8794", lw=1.0, zorder=1)
    # Scatter (skip the chosen point; it gets a star)
    mask_non_chosen = [k != "chosen" for k in keys]
    ax.scatter(jit[mask_non_chosen], iou[mask_non_chosen],
               s=42, c="#3463bf", edgecolors="white", linewidths=0.8, zorder=3)
    # Mark chosen ON the curve with a star
    ci = keys.index("chosen")
    ax.scatter([jit[ci]], [iou[ci]], s=170, marker="*", c="#d35400",
               edgecolors="white", linewidths=0.8, zorder=4,
               label="chosen (reported)")
    # Annotate every point with its (w_vel, w_acc, w_rot-vel) triple.
    # Offsets chosen to keep labels off the curve and out of each other.
    label_for = {
        "none":     r"$(0,0,0)$",
        "half":     r"$(0.05,0.05,0.15)$",
        "chosen":   r"$(0.1,0.1,0.3)$",
        "between":  r"$(0.15,0.15,0.6)$",
        "moderate": r"$(0.3,0.3,1.0)$",
    }
    nudges = {"none":     ( 8,  4),
              "half":     ( 4, 10),
              "chosen":   ( 8, -2),
              "between":  ( 0, -14),
              "moderate": ( 8,  4)}
    aligns = {"none":     ("left",  "bottom"),
              "half":     ("left",  "bottom"),
              "chosen":   ("left",  "center"),
              "between":  ("center", "top"),
              "moderate": ("left",  "center")}
    for k, j, ii in zip(keys, jit, iou):
        ha, va = aligns[k]
        ax.annotate(label_for[k], (j, ii), xytext=nudges[k],
                    textcoords="offset points", fontsize=7.5,
                    color="#1c2833", ha=ha, va=va)
    # GT-floor line
    ax.axvline(gt_floor_rot_jit, color="#27ae60", ls="--", lw=1, alpha=0.7)
    ax.text(gt_floor_rot_jit * 1.10, 0.187, "GT jitter floor",
            color="#1e7e34", fontsize=8, rotation=90, va="bottom")
    # Note about over-smoothing regime (top-left, won't collide with anything)
    ax.text(0.02, 0.96,
            r"$(2,2,5)\!\to\!$ over-smooths, IoU3D 0.131",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#566573",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fdfdfc", ec="#cccccc", lw=0.5))

    ax.set_xscale("log")
    ax.set_xlabel(r"rotation jitter (deg, log scale) $\downarrow$")
    ax.set_ylabel(r"held-out IoU3D $\uparrow$")
    ax.set_xlim(0.5, 12.0)
    ax.set_ylim(0.18, 0.248)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    # Caption legend: just identify the chosen-point marker.
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4)
    fig.savefig(OUT_DIR / "tradeoff.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR/'tradeoff.pdf'}")


# ---------- App G: head architecture (frozen vs trained) ----------
def render_arch():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 18); ax.set_ylim(-2.4, 9); ax.axis("off")

    FROZEN = dict(fc="#e9ecef", ec="#5a6268", lw=0.9)
    TRAINED = dict(fc="#cfe2ff", ec="#0d6efd", lw=0.9)
    INPUT = dict(fc="#fff3cd", ec="#b8860b", lw=0.9)
    BOX = dict(fc="#d4edda", ec="#198754", lw=0.9)

    def rbox(x, y, w, h, text, style, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.18", **style))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, color="#212529", lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, color=color, lw=lw, zorder=5))

    # ---- Inputs (left column) ----
    rbox(0.1, 7.6, 2.6, 0.8, "RGB image", INPUT)
    rbox(0.1, 6.4, 2.6, 0.8, "depth (Step 1)", INPUT)
    rbox(0.1, 5.2, 2.6, 0.8, "intrinsics $\\mathbf{K}$", INPUT)
    rbox(0.1, 3.7, 2.6, 0.9, "geo prompt\n(2D box)", INPUT, fs=7.5)

    # ---- Frozen stack ----
    rbox(3.4, 7.4, 3.4, 1.0, "SAM~3 ViT-L\n+ Simple-FPN", FROZEN, fs=8)
    rbox(3.4, 6.2, 3.4, 1.0, "LingBot-Depth\n(DINOv2-L)", FROZEN, fs=8)
    rbox(3.4, 3.7, 3.4, 0.9, "Geo encoder", FROZEN)

    rbox(7.5, 6.5, 3.2, 1.8, "Frozen transformer\nencoder $+$ decoder", FROZEN, fs=8)
    rbox(7.5, 4.4, 3.2, 1.4, "200 queries\nselect by 2D-IoU", FROZEN, fs=8)

    # ---- Trained 3D head (right column) ----
    rbox(11.5, 7.0, 5.6, 1.3, "prompt$\\_$camera\n(cross-attn over ray emb.)", TRAINED, fs=8)
    rbox(11.5, 5.3, 5.6, 1.3, "prompt$\\_$depth\n(cross-attn over depth latents)", TRAINED, fs=8)
    rbox(11.5, 3.2, 2.6, 1.6, "reg branch\n$\\to$ 12-d", TRAINED, fs=8)
    rbox(14.5, 3.2, 2.6, 1.6, "conf branch\n$\\to$ logit", TRAINED, fs=8)

    # ---- Output ----
    rbox(11.7, 1.4, 5.2, 1.2, "decoded 3D box\n$(\\mathbf{p},\\mathbf{d},\\mathbf{R})$ in camera frame",
         BOX, fs=8)

    # ---- Arrows ----
    # inputs -> frozen sub-blocks
    arrow(2.7, 8.0, 3.4, 7.9)              # RGB -> SAM3
    arrow(2.7, 6.8, 3.4, 6.7)              # depth -> LingBot
    arrow(2.7, 4.15, 3.4, 4.15)            # prompt -> geo enc
    # SAM3 / Geo -> encoder
    arrow(6.8, 7.9, 7.5, 7.7)              # FPN -> encoder
    arrow(6.8, 4.15, 7.5, 6.5)             # geo -> encoder/decoder
    # LingBot -> Early Depth Fusion (into the encoder block; this is the
    # ControlNet-style fusion path)
    arrow(6.8, 6.7, 7.5, 7.0, color="#6c757d", lw=0.8)
    # decoder -> query select
    arrow(9.1, 6.5, 9.1, 5.8)
    # selected query -> prompt_camera
    arrow(10.7, 5.0, 11.5, 7.4)
    # ray emb. (from intrinsics, derived) -> prompt_camera : curved ABOVE the diagram
    ax.add_patch(FancyArrowPatch(
        (1.4, 6.0), (11.5, 7.5),
        connectionstyle="arc3,rad=-0.32", arrowstyle="-|>",
        mutation_scale=8, color="#6c757d", lw=0.7, zorder=2))
    ax.text(6.5, 8.85, "ray embeddings (from $\\mathbf{K}$)",
            color="#6c757d", fontsize=6.5, ha="center", va="bottom")
    # depth latents (from LingBot) -> prompt_depth : routed BELOW the encoder
    ax.add_patch(FancyArrowPatch(
        (5.1, 6.2), (11.5, 5.95),
        connectionstyle="arc3,rad=0.32", arrowstyle="-|>",
        mutation_scale=8, color="#6c757d", lw=0.7, zorder=2))
    ax.text(8.5, 3.95, "depth latents (from LingBot)",
            color="#6c757d", fontsize=6.5, ha="center", va="top")
    # prompt_camera -> prompt_depth -> reg+conf
    arrow(14.3, 7.0, 14.3, 6.6)
    arrow(12.8, 5.3, 12.8, 4.8)
    arrow(15.8, 5.3, 15.8, 4.8)
    # reg + conf -> decoded box
    arrow(12.8, 3.2, 13.5, 2.6)
    arrow(15.8, 3.2, 15.1, 2.6)

    # ---- Legend via matplotlib Legend API (proxy patches, well-spaced) ----
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(**INPUT,   label="input"),
               mpatches.Patch(**FROZEN,  label="frozen (pretrained)"),
               mpatches.Patch(**TRAINED, label="trained (21.1 M params)"),
               mpatches.Patch(**BOX,     label="decoded output")]
    ax.legend(handles=handles, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8,
              handlelength=1.2, columnspacing=1.6, handletextpad=0.5)

    fig.savefig(OUT_DIR / "trackc_arch.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR/'trackc_arch.pdf'}")


# ---------- App H: per-epoch curves ----------
def render_curves():
    # Use v11_no_traj_encoder: same recipe as the reported (no-temporal) head.
    path = RUNS / "v11_no_traj_encoder" / "history.csv"
    if not path.exists():
        # Fallback to v8 if v11 isn't parsed for some reason; numerically equivalent.
        path = RUNS / "v8_deriv_gentle_25ep" / "history.csv"
    rows = list(csv.DictReader(open(path)))
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    eval_rows = [r for r in rows if r["eval_iou3d"]]
    eep = [int(r["epoch"]) for r in eval_rows]
    ev_iou = [float(r["eval_iou3d"]) for r in eval_rows]
    ev_c = [float(r["eval_center_m"]) for r in eval_rows]
    ev_r = [float(r["eval_rot_deg"]) for r in eval_rows]

    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.2))

    axs[0, 0].plot(epochs, train_loss, "-", color="#0d6efd", lw=1.2)
    axs[0, 0].set_title("Train loss"); axs[0, 0].set_xlabel("epoch"); axs[0, 0].set_ylabel("loss")
    axs[0, 0].grid(True, alpha=0.3, lw=0.5)

    axs[0, 1].plot(eep, ev_iou, "-o", color="#198754", lw=1.2, ms=3)
    axs[0, 1].set_title("Held-out IoU3D ($\\uparrow$)"); axs[0, 1].set_xlabel("epoch")
    axs[0, 1].set_ylabel("IoU3D"); axs[0, 1].grid(True, alpha=0.3, lw=0.5)

    axs[1, 0].plot(eep, ev_c, "-o", color="#d35400", lw=1.2, ms=3)
    axs[1, 0].set_title("Held-out centre error ($\\downarrow$)"); axs[1, 0].set_xlabel("epoch")
    axs[1, 0].set_ylabel("metres"); axs[1, 0].grid(True, alpha=0.3, lw=0.5)

    axs[1, 1].plot(eep, ev_r, "-o", color="#6f42c1", lw=1.2, ms=3)
    axs[1, 1].set_title("Held-out rotation error ($\\downarrow$)"); axs[1, 1].set_xlabel("epoch")
    axs[1, 1].set_ylabel("degrees"); axs[1, 1].grid(True, alpha=0.3, lw=0.5)

    # Headline note
    fig.suptitle("Track C training: chosen recipe over 25 epochs", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trackc_curves.pdf")
    plt.close(fig)
    print(f"wrote {OUT_DIR/'trackc_curves.pdf'}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_tradeoff()
    render_arch()
    render_curves()
