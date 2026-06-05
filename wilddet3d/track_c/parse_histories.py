"""Parse Track C train.log files into clean per-epoch history CSVs + md summary."""
import re, glob, os, csv

RUNS = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs/track_c/runs"
order = ["v1", "v2_from_pretrained", "v3_no_temporal", "v4_deriv_on",
         "v5_deriv_off", "v6_deriv_gentle", "v7_deriv_moderate"]

tr = re.compile(r"\[ep (\d+)\] train_loss=([\d.]+) \(center=([\d.]+) depth=([\d.]+) "
                r"dims=([\d.]+) rot_deg=([\d.]+)\)(?: deriv\(cvel=([\d.]+) cacc=([\d.]+) "
                r"rotvel_deg=([\d.]+)\))?.*?gate\|\.\|=([\d.]+) lr=([\d.eE+-]+) (\d+)s")
ev = re.compile(r"\[ep (\d+)\] EVAL track_c: iou3d=([\d.]+) center=([\d.]+)m "
                r"dims=([\d.]+)m rot=([\d.]+)deg")

for run in order:
    log = f"{RUNS}/{run}/train.log"
    if not os.path.exists(log):
        continue
    rows = {}
    txt = open(log, errors="ignore").read()
    for m in tr.finditer(txt):
        e = int(m.group(1))
        rows[e] = dict(epoch=e, train_loss=m.group(2), tl_center=m.group(3),
                       tl_depth=m.group(4), tl_dims=m.group(5), tl_rot_deg=m.group(6),
                       d_cvel=m.group(7) or "", d_cacc=m.group(8) or "",
                       d_rotvel_deg=m.group(9) or "", gate=m.group(10),
                       lr=m.group(11), epoch_sec=m.group(12),
                       eval_iou3d="", eval_center_m="", eval_dims_m="", eval_rot_deg="")
    for m in ev.finditer(txt):
        e = int(m.group(1))
        if e in rows:
            rows[e].update(eval_iou3d=m.group(2), eval_center_m=m.group(3),
                           eval_dims_m=m.group(4), eval_rot_deg=m.group(5))
    if not rows:
        continue
    cols = ["epoch", "train_loss", "tl_center", "tl_depth", "tl_dims", "tl_rot_deg",
            "d_cvel", "d_cacc", "d_rotvel_deg", "gate", "lr", "epoch_sec",
            "eval_iou3d", "eval_center_m", "eval_dims_m", "eval_rot_deg"]
    out = f"{RUNS}/{run}/history.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in sorted(rows):
            w.writerow(rows[e])
    tls = [rows[e]["train_loss"] for e in sorted(rows)]
    print(f"{run}: {len(rows)} epochs -> history.csv ; train_loss [{', '.join(tls)}]")
