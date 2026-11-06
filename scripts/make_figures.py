#!/usr/bin/env python3
"""
Result figures for the paper, drawn from results/*.json, results/cache/figdata_*.npz and
results/additivity/*_curves.npz.

    python scripts/make_figures.py            # all
    python scripts/make_figures.py fidelity   # one
"""

import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from matplotlib.patches import Patch

from figstyle import (BLUE, DATASET_TITLE, FULL, GOLD, GREY1, GRID, INK, INK2, METHOD, MUTED,
                      RED, ROOT, TEAL, panel_label, save)

RES = ROOT / "results"
sys.path.insert(0, str(ROOT))
from explorer import protocol as P                                           # noqa: E402

DS = list(P.DATASETS)
TOP_A = f"{100 * P.TOPK_FRACS[0]:g}%"
CI_TXT = f"{100 * P.CI_LEVEL:g}% CI"
NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five"}
SUMMARY = json.loads((RES / "summary.json").read_text())


def records(ds, variant=None, set_="main"):
    variant = variant or P.REFERENCE_VARIANT[ds]
    return json.loads((RES / "fidelity" / f"{ds}_{variant}_{set_}.json").read_text())["records"]


def figdata(ds):
    return np.load(RES / "cache" / f"figdata_{ds}.npz", allow_pickle=False)


# =========================================================================== fidelity
def fig_fidelity():
    order = ["taylor", "taylor_frozen", "coupling", "act_x_coupling", "g_C", "g_A", "activation", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 3.15), sharey=True)
    y = np.arange(len(order))[::-1]
    for c, (ax, ds) in enumerate(zip(axes, DS)):
        run = SUMMARY["runs"][f"{ds}/{P.REFERENCE_VARIANT[ds]}/main"]["rank"]
        rob = SUMMARY["robustness"][ds]
        for k, yy in enumerate(y):
            if k % 2 == 0:
                ax.axhspan(yy - 0.5, yy + 0.5, color="#f6f5f2", zorder=0, lw=0)
        for m, yy in zip(order, y):
            if m not in run:
                ax.text(1.4e-3, yy, "n/a", va="center", fontsize=6.5, color=MUTED)
                continue
            lab, col, *_ = METHOD[m]
            d = run[m]["p1"]
            miss, lo, hi = 1 - d["mean"], 1 - d["mean_ci"][1], 1 - d["mean_ci"][0]
            miss, lo, hi = (max(v, 1e-3) for v in (miss, lo, hi))
            ax.plot([lo, hi], [yy, yy], color=col, lw=2.2, solid_capstyle="round", zorder=3)
            ax.scatter([miss], [yy], s=30, color=col, edgecolor="white", linewidth=0.8, zorder=4)
            # per-checkpoint means (robustness): small ticks
            per = rob.get(m, {}).get("p1", {}).get("per_checkpoint_mean", {})
            for val in per.values():
                ax.plot([max(1 - val, 1e-3)] * 2, [yy + 0.16, yy + 0.38], color=col, lw=0.9,
                        alpha=0.8, zorder=2)
            if m in ("taylor", "taylor_frozen", "coupling"):
                ax.text(min(hi * 1.35, 0.7), yy, f"{100 * d['mean']:.1f}%", va="center", fontsize=6.6,
                        color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 1.25)
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
        ax.set_xticklabels(["0.1%", "1%", "10%", "100%"])
        ax.grid(axis="y", visible=False)
        ax.set_title(DATASET_TITLE[ds], pad=4)
        ax.tick_params(axis="y", length=0)
        if c == 1:
            ax.set_xlabel(f"share of the true top-{TOP_A} capsules missed  (1 − P@{TOP_A}, mean and {CI_TXT}, log scale)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([METHOD[m][0] for m in order])
    for t in axes[0].get_yticklabels():
        t.set_color(INK)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    handles = [Line2D([], [], marker="o", color=INK2, lw=0, markersize=4, label=f"most accurate run (mean, {CI_TXT}); numbers give P@{TOP_A} recovered"),
               Line2D([], [], color=INK2, lw=0.8, marker="|", markersize=7, linestyle="none",
                      label=f"{NUM_WORD.get(len(P.SEED_VARIANTS), len(P.SEED_VARIANTS))} runs (means)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.6, 1.05))
    fig.tight_layout(w_pad=1.6)          # room between one panel's "100%" and the next panel's "0.1%"
    save(fig, "fig_fidelity")


# =========================================================================== routing dynamics
def fig_routing_dynamics():
    rows = [("consensus_iter", "consensus\n(share backing $\\hat\\jmath$)"),
            ("entropy_top1_iter", f"routing entropy / $\\log K$\n(top-{TOP_A} capsules)"),
            ("vlen_iter", r"winning length $\|v_{\hat\jmath}\|$")]
    fig, axes = plt.subplots(3, 3, figsize=(FULL, 4.6), sharex=True)
    n_iter = len(records(DS[0])[0]["vlen_iter"])
    x = np.arange(1, n_iter + 1)
    for c, ds in enumerate(DS):
        main, mis = records(ds), records(ds, set_="misclassified")
        for r, (key, lab) in enumerate(rows):
            ax = axes[r, c]
            arr = np.array([q[key] for q in main])
            for line in arr:
                ax.plot(x, line, color=BLUE, alpha=0.035, lw=0.7, zorder=1)
            ax.plot(x, np.median(arr, 0), color=INK, lw=1.8, marker="o", markersize=3.5, zorder=4,
                    label="median, correct + incorrect (main set)")
            marr = np.array([q[key] for q in mis])
            ax.plot(x, np.median(marr, 0), color=RED, lw=1.4, ls="--", marker="o", markersize=3,
                    zorder=4, label="median, misclassified set")
            if key == "entropy_top1_iter":
                lo = np.concatenate([arr.ravel(), marr.ravel()]).min()     # show every image
                ax.set_ylim(max(0, lo - 0.01), 1.003)
            if key == "consensus_iter":
                ax.set_ylim(-0.02, 1.0)
            if key == "vlen_iter":
                ax.set_ylim(0, 1.0)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
            if r == 0:
                ax.set_title(DATASET_TITLE[ds], pad=4)
            if r == 2:
                ax.set_xlabel("routing iteration")
            ax.set_xticks(x)
            ax.set_xlim(0.85, n_iter + 0.15)
    # True by construction: routing logits start at zero, so iteration-1 couplings are uniform.
    c1 = SUMMARY["runs"][f"{DS[0]}/{P.REFERENCE_VARIANT[DS[0]]}/main"]["routing"]["consensus_iter"]["mean"][0]
    axes[0, 0].annotate("iteration 1: logits $b=0$,\ncouplings uniform", xy=(1.03, c1 + 0.02), xytext=(1.12, 0.78),
                        fontsize=6.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=2))
    n_main = len(records(DS[0]))
    handles = [Line2D([], [], color=BLUE, alpha=0.5, lw=1, label=f"one image (main set, {n_main} per dataset)"),
               Line2D([], [], color=INK, lw=1.8, marker="o", markersize=3.5, label="median, main set"),
               Line2D([], [], color=RED, lw=1.4, ls="--", marker="o", markersize=3, label="median, misclassified set")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 1.03))
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    for k, ax in enumerate(axes[:, 0]):
        panel_label(ax, "abc"[k], x=-0.36, y=1.0)
    save(fig, "fig_routing_dynamics")


# =========================================================================== decision support
def fig_decision_support():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.3), sharey=True)
    specs = [("routing_share", "relative size of the routing term in $\\partial\\varphi/\\partial u$", BLUE),
             ("concentration_top1", f"share of positive effect held by the top {TOP_A}", TEAL)]
    rng = np.random.default_rng(0)
    for ax, (key, lab, col) in zip(axes, specs):
        for k, ds in enumerate(DS[::-1]):
            v = np.array([q[key] for q in records(ds)])
            yy = k + rng.uniform(-0.22, 0.22, len(v))
            ax.scatter(v, yy, s=3, color=col, alpha=0.28, lw=0, zorder=2)
            q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
            ax.plot([q1, q3], [k, k], color=INK, lw=3.2, solid_capstyle="butt", zorder=3, alpha=0.85)
            ax.scatter([med], [k], s=34, color="white", edgecolor=INK, lw=1.2, zorder=4)
            per = SUMMARY["robustness"][ds][key]["per_checkpoint_median"]
            seeds = list(per.values())
            ax.scatter(seeds, [k + 0.36] * len(seeds), marker="v", s=14,
                       color=INK2, zorder=4, lw=0)
            ax.text(med, k - 0.42, f"{med:.2f}", ha="center", va="top", fontsize=6.8, color=INK)
        ax.set_xlim(0, 1)
        ax.set_xlabel(lab)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels([DATASET_TITLE[d] for d in DS[::-1]])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.7, 2.6)
    handles = [Line2D([], [], marker="o", color=INK2, lw=0, markersize=2, alpha=0.6, label="one image"),
               Line2D([], [], color=INK, lw=3, label="interquartile range"),
               Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK, lw=0, label="median"),
               Line2D([], [], marker="v", color=INK2, lw=0, markersize=4, label="median of each run")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.55, 1.12))
    fig.tight_layout(w_pad=1.5)
    panel_label(axes[0], "a", x=-0.2, y=0.98)
    panel_label(axes[1], "b", x=-0.04, y=0.98)
    save(fig, "fig_decision_support")


# =========================================================================== deletion
def fig_deletion():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.55), sharey=True)
    show = [("activation", 1.0, False), ("g_A", 1.3, False), ("g_C", 1.5, True),
            ("taylor", 2.2, True), ("exact", 1.4, True)]
    for c, (ax, ds) in enumerate(zip(axes, DS)):
        d = figdata(ds)
        N = int(d["N"])
        k = d["ks"] * N
        sel = k >= 1
        for m, lw, band in show:
            lab, col, _, ls, _ = METHOD[m]
            y = d[f"curve_{m}"][:, sel]
            med = np.median(y, 0)
            if band:
                ax.fill_between(k[sel], *np.quantile(y, [0.25, 0.75], 0), color=col, alpha=0.10, lw=0)
            # activation is dotted here so that it is not confused with the dashed exact curve
            ax.plot(k[sel], med, color=col, lw=lw, ls=":" if m == "activation" else ls, label=lab,
                    zorder=5 if m == "exact" else 4)
        ex = np.median(d["curve_exact"][:, sel], 0)
        kk = k[sel]
        part = kk < N
        i_min = int(np.argmin(ex[part]))
        i_max = i_min + int(np.argmax(ex[part][i_min:]))
        ax.annotate("collapse", xy=(kk[i_min], ex[i_min]), xytext=(kk[i_min] * 0.18, 0.06),
                    fontsize=7, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.6))
        ax.annotate("", xy=(kk[i_max], ex[i_max]), xytext=(kk[i_max], ex[i_min]),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
        # the rebound label sits in a band above the curves, joined to the arrow by a leader
        ax.plot([kk[i_max]] * 2, [ex[i_max] + 0.03, 1.1], color=RED, lw=0.5, ls=(0, (1, 1.5)))
        ax.text(kk[i_max], 1.12, f"rebound +{ex[i_max] - ex[i_min]:.2f}", fontsize=7, color=INK,
                va="bottom", ha="right" if c < 2 else "center")
        ax.set_xscale("log")
        ax.set_xlim(1, N)
        ax.set_ylim(0, 1.24)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.set_title(f"{DATASET_TITLE[ds]}  (N = {N:,})", pad=4)
        if c == 1:
            ax.set_xlabel("capsules removed together, in each method's ranking order (log scale)")
    axes[0].set_ylabel(r"confidence kept  $\varphi/\varphi_0$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.07))
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_deletion")


# =========================================================================== additivity
def fig_additivity():
    """Deletion in descending Δ with routing re-run and with the intact couplings held fixed,
    against the sum of single effects (PROTOCOL.md, "Group deletion against additivity"), on the additivity set. Top:
    confidence (median, IQR); bottom: the interactions I_R and I_F (median)."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.5), sharey="row", sharex="col",
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
    show = [("additive", r"sum of the single effects $\Delta_i$", GREY1, "--", 1.3, False),
            ("frozen", "couplings held fixed", GOLD, "-", 1.4, True),
            ("route", "routing re-run", INK, "-", 1.6, True)]
    inter = [("I_F", GOLD, 1.4), ("I_R", INK, 1.6)]
    for c, ds in enumerate(DS):
        z = np.load(RES / "additivity" / f"{ds}_{P.REFERENCE_VARIANT[ds]}_additivity_curves.npz", allow_pickle=False)
        names = list(z["names"])
        k = z["grid"]
        N = int(k[-1])
        sel = k >= 1
        A = SUMMARY["additivity"][ds]
        kp = A["points"]["kplus"]["k"]
        km = A["by_tau"]["1.0"]["kminus_frac"]["median"] * N
        for ax in axes[:, c]:
            ax.axvspan(kp["q25"], kp["q75"], color=GRID, alpha=0.7, lw=0, zorder=1)
            ax.axvline(kp["median"], color=INK2, lw=0.8, ls=":", zorder=2)
            ax.axvline(km, color=INK2, lw=0.8, ls="-.", zorder=2)
        ax = axes[0, c]
        for name, lab, col, ls, lw, band in show:
            y = z["curves"][:, names.index(name), :][:, sel]
            if band:
                ax.fill_between(k[sel], *np.quantile(y, [0.25, 0.75], 0), color=col, alpha=0.12, lw=0, zorder=3)
            ax.plot(k[sel], np.median(y, 0), color=col, lw=lw, ls=ls, label=lab, zorder=4)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"{DATASET_TITLE[ds]}  (N = {N:,})", pad=4)
        ax = axes[1, c]
        ax.axhline(0, color=MUTED, lw=0.6, zorder=2)
        for name, col, lw in inter:
            ax.plot(k[sel], np.median(z["curves"][:, names.index(name), :][:, sel], 0), color=col, lw=lw, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(1, N)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        if c == 1:
            ax.set_xlabel(r"capsules removed together, in descending order of $\Delta$ (log scale)")
    axes[0, 0].set_ylabel(r"confidence kept  $\varphi/\varphi_0$")
    axes[1, 0].set_ylabel(r"interaction  $I/\varphi_0$ (median)")
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [(Patch(facecolor=GRID, lw=0), Line2D([], [], color=INK2, lw=0.8, ls=":")),
          Line2D([], [], color=INK2, lw=0.8, ls="-.")]
    l += [r"$k_+$, last capsule with $\Delta>\delta$", r"$k_-$, first capsule with $\Delta<-\delta$"]
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.07))
    fig.tight_layout(w_pad=1.2, h_pad=0.6)
    save(fig, "fig_additivity")


# =========================================================================== capsule gates
def fig_gates():
    """Capsule gates f_i(α) of the gate-analysis images (experiments/run_gates.py): the
    response, normalised by Δ_i, and its slope. Δ_i is the mean slope over [0, 1] and the
    Taylor estimate the slope at α = 1."""
    gates = json.loads((RES / "gates.json").read_text())
    groups = [("rank_k", f"ranks $k$, $k+1$ (top-{TOP_A} cutoff)", BLUE), ("rank_k1", None, BLUE),
              ("rank_topset", f"rank {100 * P.TOP_SET_FRAC:g}% of $N$", TEAL), ("rank1", "rank 1", INK)]
    alpha = np.linspace(0, 1, P.GATE_STEPS)
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 4.2), sharex=True, sharey="row")
    for c, ds in enumerate(DS):
        for g, lab, col in groups:
            for n, r in enumerate(gates[ds]["rows"]):
                d = r["gates"][g]
                f = np.asarray(d["f"])
                axes[0, c].plot(alpha, (f[-1] - f) / d["exact"], color=col, lw=0.7, alpha=0.55,
                                label=lab if n == 0 else None)
                axes[1, c].plot(alpha, np.gradient(f, alpha, edge_order=2) / d["exact"], color=col, lw=0.7, alpha=0.55)
        axes[0, c].plot([0, 1], [1, 0], color=GREY1, lw=0.9, ls="--", label="straight gate", zorder=5)
        axes[1, c].axhline(1, color=GREY1, lw=0.9, ls="--", zorder=5)
        axes[0, c].set_title(DATASET_TITLE[ds], pad=4)
        if c == 1:
            axes[1, c].set_xlabel(r"gate $\alpha$ on capsule $i$ (0 removed, 1 intact)")
    axes[0, 0].set_ylabel(r"$[f_i(1)-f_i(\alpha)]/\Delta_i$")
    axes[1, 0].set_ylabel(r"$f_i'(\alpha)/\Delta_i$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.05))
    fig.tight_layout(w_pad=1.2, h_pad=0.6)
    save(fig, "fig_gates")


# =========================================================================== magnitude
def fig_magnitude():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.55))
    for c, (ax, ds) in enumerate(zip(axes, DS)):
        d = figdata(ds)
        mag = SUMMARY["runs"][f"{ds}/{P.REFERENCE_VARIANT[ds]}/main"]["magnitude"]
        ex = d["mag_exact"]
        # the two series differ in marker shape as well as hue (filled dots vs crosses)
        for m, col, z, mk in (("taylor_frozen", GOLD, 2, "x"), ("taylor", BLUE, 3, "o")):
            est = d[f"mag_{m}"]
            ok = (ex > 0) & (est > 0)
            if mk == "o":
                ax.scatter(ex[ok], est[ok], s=2.2, color=col, alpha=0.32, lw=0, zorder=z, rasterized=True)
            else:
                ax.scatter(ex[ok], est[ok], s=3.2, color=col, alpha=0.45, lw=0.45, marker=mk, zorder=z,
                           rasterized=True)
            sl = mag[m]["slope_top5"]["median"]
            yk = 0.93 if m == "taylor" else 0.84
            ax.scatter([0.05], [yk], s=16, color=col, lw=0 if mk == "o" else 1.2, marker=mk,
                       transform=ax.transAxes, clip_on=False, zorder=7)
            ax.text(0.09, yk, ("through routing" if m == "taylor" else "routing frozen") + f":  slope {sl:.2f}",
                    transform=ax.transAxes, color=INK, fontsize=7.2, va="center")
        lo, hi = ex[ex > 0].min(), max(ex.max(), d["mag_taylor"].max())
        # every drawable (positive) estimate stays inside the axes: nothing is clipped silently
        est_lo = min(d[f"mag_{m}"][(ex > 0) & (d[f"mag_{m}"] > 0)].min() for m in ("taylor", "taylor_frozen"))
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls=(0, (4, 3)), zorder=6)
        ax.text(0.96, 0.06, "dashed: $y = x$", transform=ax.transAxes, ha="right", fontsize=6.6, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo * 0.8, hi * 1.4); ax.set_ylim(min(lo, est_lo) * 0.6, hi * 3)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.set_title(DATASET_TITLE[ds], pad=4)
        ax.set_xlabel(r"exact ablation effect $\Delta_i$")
        if c == 0:
            ax.set_ylabel(r"first-order estimate $\hat\Delta_i$")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_magnitude")


# =========================================================================== cost
def fig_cost():
    cost = json.loads((RES / "cost.json").read_text())["results"]
    rows = [("forward", "forward pass", GREY1), ("gradient_attributions", "gradient attribution", BLUE),
            ("exact_ablation", "exact ablation", INK)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.75), sharey=True)
    y = np.arange(len(rows))[::-1]
    for c, (ax, ds) in enumerate(zip(axes, DS)):
        ax.axvspan(0.1, P.INTERACTIVE_MS, color="#eef5ec", lw=0, zorder=0)
        for (key, lab, col), yy in zip(rows, y):
            g, cpu = cost[ds]["cuda"][key]["median_ms"], cost[ds]["cpu"][key]["median_ms"]
            ax.plot([g, cpu], [yy, yy], color=col, lw=1.0, alpha=0.45, zorder=2)
            ax.scatter([g], [yy], s=24, color=col, zorder=3, lw=0)
            ax.scatter([cpu], [yy], s=24, facecolor="white", edgecolor=col, lw=1.2, zorder=3)
            if key in ("gradient_attributions", "exact_ablation"):
                txt = f"{cpu / 1000:.1f} s" if cpu >= 1000 else f"{cpu:.0f} ms"
                ax.text(max(g, cpu) * 1.7, yy, f"CPU {txt}", va="center", fontsize=6.4, color=INK2)
        ax.set_xscale("log"); ax.set_xlim(0.1, 3e5)
        ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5]); ax.set_xticklabels(["1", "10", "100", "1k", "10k", "100k"])
        ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
        ax.set_title(DATASET_TITLE[ds], pad=4)
        if c == 1:
            ax.set_xlabel("time for one test image in ms (median of repeated runs, log scale)")
    axes[0].set_yticks(y); axes[0].set_yticklabels([r[1] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.2)
    gpu = json.loads((RES / "cost.json").read_text())["provenance"]["device"].replace("NVIDIA GeForce ", "")
    handles = [Line2D([], [], marker="o", color=INK2, lw=0, label=f"GPU ({gpu})"),
               Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK2, lw=0, label="CPU only"),
               Patch(facecolor="#eef5ec", edgecolor="none", label=f"interactive (< {P.INTERACTIVE_MS} ms)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.1))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_cost")


# =========================================================================== sanity
def fig_sanity():
    stages = ["trained"] + json.loads((RES / "sanity" / f"{DS[0]}.json").read_text())["provenance"]["stages"]
    show = ["exact", "taylor", "g_C", "coupling", "activation"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), sharey=True)
    for c, (ax, ds) in enumerate(zip(axes, DS)):
        s = json.loads((RES / "sanity" / f"{ds}.json").read_text())["median_similarity"]
        x = np.arange(len(stages))
        dodge = dict(zip(show, np.linspace(-0.12, 0.12, len(show))))   # display offset only
        for m in show:
            lab, col, _, ls, mk = METHOD[m]
            y = [1.0] + [s[st][m] for st in stages[1:]]
            ax.plot(x + dodge[m], y, color=col, ls=ls, marker=mk, markersize=4, lw=1.3,
                    label=lab, zorder=4 if m == "taylor" else 3)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(stages, rotation=18, ha="right")
        ax.set_ylim(-0.25, 1.08)
        ax.set_title(DATASET_TITLE[ds], pad=24)
        tvx = s[stages[-1]]["taylor_vs_exact_randomized"]
        ax.text(0.5, 1.015, f"Taylor vs exact after the full\ncascade: $\\rho$ = {tvx:.2f}", ha="center",
                va="bottom", transform=ax.transAxes, fontsize=6.8, color=INK2, linespacing=1.1)
    axes[0].set_ylabel("Spearman $\\rho$ with the\ntrained model's attribution")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.13))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_sanity")


FIGS = {"fidelity": fig_fidelity, "routing": fig_routing_dynamics, "support": fig_decision_support,
        "deletion": fig_deletion, "additivity": fig_additivity, "gates": fig_gates, "magnitude": fig_magnitude, "cost": fig_cost, "sanity": fig_sanity}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        print(name)
        FIGS[name]()
