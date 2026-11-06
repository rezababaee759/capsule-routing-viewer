#!/usr/bin/env python3
"""
Case studies (rule-selected in scripts/figure_data.py): for each dataset, the most
confident correct prediction, the most confident error, and the correct prediction
with the smallest top-2 margin. Each grid cell's positive effect (summed over capsule
types) is drawn as a Gaussian centred on its receptive field to give a capsule
attribution projection; exact ablation and the Taylor estimate are shown side by side.
"""

import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from figstyle import DATASET_TITLE, FULL, GOLD, INK, INK2, ROOT, save

import sys
sys.path.insert(0, str(ROOT))
from explorer import attribution as AT                                      # noqa: E402
from explorer import protocol as P                                          # noqa: E402
from explorer.registry import DATASETS, geometry_of                         # noqa: E402

CASES = [("confident_correct", "most confident correct"), ("ambiguous", "smallest margin (correct)"),
         ("confident_error", "most confident error")]
WARM = LinearSegmentedColormap.from_list("warm", ["#ffd27a", GOLD, "#b2341f"])


SIGMA_FRAC = 0.25        # Gaussian width as a fraction of the receptive field (display choice)


def evidence(geo, delta):
    """Pixel-space evidence: each grid cell's positive effect as a Gaussian (σ = SIGMA_FRAC·rf)
    centred on its receptive field (a display choice; effective receptive fields are centre-weighted)."""
    sigma = SIGMA_FRAC * geo.rf
    S = geo.img_size
    yy, xx = np.mgrid[0:S, 0:S]
    E = np.zeros((S, S))
    pos = np.clip(delta, 0, None)
    per_cell = geo.to_grid(pos).sum(0)                     # [G, G], summed over types
    for gy in range(geo.G):
        for gx in range(geo.G):
            if per_cell[gy, gx] <= 0:
                continue
            cx, cy = geo.rf_center(gx, gy)
            E += per_cell[gy, gx] * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return E


def overlay(ax, img, E, vmax):
    base = np.asarray(img)
    base = np.repeat(base[..., None], 3, -1) if base.ndim == 2 else base
    grey = base.mean(-1, keepdims=True)
    ax.imshow(np.clip(0.25 * base + 0.6 * grey, 0, 1), interpolation="nearest")   # muted base
    v = np.clip(E / vmax, 0, 1)
    rgba = WARM(v)
    rgba[..., 3] = v ** 1.1 * 0.9
    ax.imshow(rgba, interpolation="bicubic")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig = plt.figure(figsize=(FULL, FULL * 0.74))
    outer = fig.add_gridspec(3, 3, hspace=0.36, wspace=0.08, left=0.045, right=0.995, top=0.93, bottom=0.06)
    for r, key in enumerate(DATASETS):
        spec = DATASETS[key]
        model = spec.build_model(device)
        geo = geometry_of(model)
        ds = spec.dataset()
        cases = json.loads(str(np.load(ROOT / "results" / "cache" / f"figdata_{key}.npz")["cases"]))
        for c, (case, title) in enumerate(CASES):
            idx = cases[case]
            x, label = ds[idx]
            res = AT.explain(model, x.unsqueeze(0).to(device))
            img = spec.to_display(x)
            Ex, Et = evidence(geo, res["exact"]), evidence(geo, res["taylor"])
            vmax = max(Ex.max(), 1e-12)
            inner = outer[r, c].subgridspec(1, 2, wspace=0.04)
            for k, (E, lab) in enumerate([(Ex, "exact"), (Et, "Taylor")]):
                ax = fig.add_subplot(inner[0, k])
                overlay(ax, img, E, vmax)
                ax.set_title(lab, fontsize=7.2, color=INK2, pad=2)
            ag = AT_agree(res)
            cls = spec.classes
            ok = res["pred"] == label
            txt = (f"{cls[label]} → {cls[res['pred']]}\n"
                   f"$\\|v_{{\\hat\\jmath}}\\|$ = {res['phi']:.2f}  ·  top-{100 * P.TOPK_FRACS[0]:g}% recovered {100 * ag:.0f}%")
            fig.text((outer[r, c].get_position(fig).x0 + outer[r, c].get_position(fig).x1) / 2,
                     outer[r, c].get_position(fig).y0 - 0.008, txt, ha="center", va="top", fontsize=6.8,
                     linespacing=1.35,
                     color=INK if ok else "#c2414b")
            if r == 0:
                fig.text((outer[r, c].get_position(fig).x0 + outer[r, c].get_position(fig).x1) / 2, 0.985,
                         title, ha="center", va="top", fontsize=9.5, color=INK)
        fig.text(0.012, (outer[r, 0].get_position(fig).y0 + outer[r, 0].get_position(fig).y1) / 2,
                 DATASET_TITLE[key], rotation=90, ha="center", va="center", fontsize=9.5, color=INK)
    save(fig, "fig_cases")


def AT_agree(res):
    ex, ty = res["exact"], res["taylor"]
    k = int(np.ceil(P.TOPK_FRACS[0] * len(ex)))
    return len(set(np.argsort(-ty)[:k]) & set(np.argsort(-ex)[:k])) / k


if __name__ == "__main__":
    main()
