#!/usr/bin/env python3
"""
Every number printed in the manuscript, generated from the repository's artifacts.

Writes
  paper/generated/numbers.tex   \\V{key} definitions (an undefined key is a LaTeX error)
  paper/generated/numbers.json  key -> {value, text, source, path, note}

Each entry records the file it was read from and the path inside it, so any value in
the PDF can be traced back without reading this script. Nothing here is typed by
hand except formatting choices.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from explorer import protocol as P                                           # noqa: E402
from explorer import attribution as AT                                       # noqa: E402
from explorer.registry import DATASETS, geometry_of                         # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "paper" / "generated"
NUMBERS = {}


# ------------------------------------------------------------------ formatting
def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def dec(x, d=3):
    t = f"{x:.{d}f}"
    return t[1:] if t.startswith("-") and float(t) == 0 else t        # no "-0.000"


def signed(x, d=3):
    t = f"{x:+.{d}f}"
    return t[1:] if float(t) == 0 else t                              # no "+0.000" / "-0.000"


def intc(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def sci(x, d=1):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if f"{m:.{d}f}" == "10." + "0" * d:
        m, e = 1.0, e + 1
    return f"{m:.{d}f}\\times10^{{{e}}}"


def ms(x):
    return f"{x:,.0f}".replace(",", "{,}") if x >= 10 else f"{x:.1f}"


def put(key, value, text, source, path="", note=""):
    if key in NUMBERS:
        raise KeyError(f"duplicate number key {key}")
    NUMBERS[key] = {"value": value if not isinstance(value, (np.floating, np.integer)) else value.item(),
                    "text": text, "source": str(source), "path": path, "note": note}


def get(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


def load(rel):
    return json.loads((ROOT / rel).read_text())


# ------------------------------------------------------------------ protocol constants
def protocol_numbers():
    src = "explorer/protocol.py"
    put("prot.seed", P.SEED, str(P.SEED), src, "SEED")
    put("prot.nperclass", P.N_PER_CLASS, str(P.N_PER_CLASS), src, "N_PER_CLASS")
    put("prot.nmain", P.N_PER_CLASS * 10, str(P.N_PER_CLASS * 10), src, "N_PER_CLASS*10")
    put("prot.misclassmax", P.MISCLASSIFIED_MAX, str(P.MISCLASSIFIED_MAX), src, "MISCLASSIFIED_MAX")
    put("prot.pilotn", P.PILOT_N, str(P.PILOT_N), src, "PILOT_N")
    put("prot.topkA", P.TOPK_FRACS[0], pct(P.TOPK_FRACS[0], 0), src, "TOPK_FRACS[0]")
    put("prot.topkB", P.TOPK_FRACS[1], pct(P.TOPK_FRACS[1], 0), src, "TOPK_FRACS[1]")
    put("prot.topset", P.TOP_SET_FRAC, pct(P.TOP_SET_FRAC, 0), src, "TOP_SET_FRAC")
    for i, b in enumerate(P.BUDGETS):
        put(f"prot.budget{i + 1}", b, pct(b, 0), src, f"BUDGETS[{i}]")
    put("prot.curvesteps", P.CURVE_STEPS, str(P.CURVE_STEPS), src, "CURVE_STEPS")
    put("prot.nboot", P.N_BOOT, intc(P.N_BOOT), src, "N_BOOT")
    put("prot.cilevel", P.CI_LEVEL, pct(P.CI_LEVEL, 0), src, "CI_LEVEL")
    put("prot.bcaskew", P.BCA_SKEW_THRESHOLD, f"{P.BCA_SKEW_THRESHOLD:g}", src, "BCA_SKEW_THRESHOLD")
    put("prot.alpha", P.ALPHA, f"{P.ALPHA:g}", src, "ALPHA")
    put("prot.ntests", len(P.CONFIRMATORY) * len(P.DATASETS), str(len(P.CONFIRMATORY) * len(P.DATASETS)),
        src, "len(CONFIRMATORY)*len(DATASETS)")
    put("prot.ncheckpoints", len(P.VARIANTS), str(len(P.VARIANTS)), src, "len(VARIANTS)")
    put("prot.nseeds", len(P.SEED_VARIANTS), str(len(P.SEED_VARIANTS)), src, "len(SEED_VARIANTS)")
    seeds = [v.replace("seed", "") for v in P.SEED_VARIANTS]
    put("prot.trainseeds", seeds, ", ".join(seeds[:-1]) + " and " + seeds[-1], src, "SEED_VARIANTS")
    put("prot.ndatasets", len(P.DATASETS), str(len(P.DATASETS)), src, "len(DATASETS)")
    put("prot.nmodels", len(P.DATASETS) * len(P.VARIANTS), str(len(P.DATASETS) * len(P.VARIANTS)), src,
        "len(DATASETS)*len(VARIANTS)")
    put("prot.randperclass", P.RANDOMIZATION_PER_CLASS, str(P.RANDOMIZATION_PER_CLASS), src,
        "RANDOMIZATION_PER_CLASS")
    put("prot.nrand", P.RANDOMIZATION_PER_CLASS * 10, str(P.RANDOMIZATION_PER_CLASS * 10), src,
        "RANDOMIZATION_PER_CLASS*10")
    put("prot.costreps", P.COST_REPS_GPU, str(P.COST_REPS_GPU), src, "COST_REPS_GPU")
    put("prot.costwarm", P.COST_WARMUP_GPU, str(P.COST_WARMUP_GPU), src, "COST_WARMUP_GPU")
    put("prot.costrepscpu", P.COST_REPS_CPU, str(P.COST_REPS_CPU), src, "COST_REPS_CPU")
    put("prot.costwarmcpu", P.COST_WARMUP_CPU, str(P.COST_WARMUP_CPU), src, "COST_WARMUP_CPU")
    put("prot.costrepsexactgpu", P.COST_REPS_EXACT_GPU, str(P.COST_REPS_EXACT_GPU), src, "COST_REPS_EXACT_GPU")
    put("prot.costrepsexactcpu", P.COST_REPS_EXACT_CPU, str(P.COST_REPS_EXACT_CPU), src, "COST_REPS_EXACT_CPU")
    put("prot.misseed", P.SEED + P.MISCLASSIFIED_SEED_OFFSET, str(P.SEED + P.MISCLASSIFIED_SEED_OFFSET), src,
        "SEED + MISCLASSIFIED_SEED_OFFSET")
    put("prot.interactivems", P.INTERACTIVE_MS, str(P.INTERACTIVE_MS), src, "INTERACTIVE_MS")
    put("prot.routemin", P.ROUTE_DRAW_MIN, f"{P.ROUTE_DRAW_MIN:g}", src, "ROUTE_DRAW_MIN")
    put("prot.gaten", P.GATE_N_PER_DATASET, str(P.GATE_N_PER_DATASET), src, "GATE_N_PER_DATASET")
    put("prot.figcurves", P.FIG_CURVES_PER_CLASS * 10, str(P.FIG_CURVES_PER_CLASS * 10), src,
        "FIG_CURVES_PER_CLASS*10")
    put("prot.figcurvesper", P.FIG_CURVES_PER_CLASS, str(P.FIG_CURVES_PER_CLASS), src, "FIG_CURVES_PER_CLASS")
    put("prot.figmagper", P.FIG_MAGNITUDE_PER_CLASS, str(P.FIG_MAGNITUDE_PER_CLASS), src, "FIG_MAGNITUDE_PER_CLASS")
    put("prot.gateseed", P.GATE_PICK_SEED, str(P.GATE_PICK_SEED), src, "GATE_PICK_SEED")
    put("prot.figmag", P.FIG_MAGNITUDE_PER_CLASS * 10, str(P.FIG_MAGNITUDE_PER_CLASS * 10), src,
        "FIG_MAGNITUDE_PER_CLASS*10")
    put("mag.farfactor", P.FAR_FACTOR, f"{P.FAR_FACTOR:g}", src, "FAR_FACTOR")
    put("prot.addtolfactor", P.ADDITIVITY_TOL_FACTOR, f"{P.ADDITIVITY_TOL_FACTOR:g}", src, "ADDITIVITY_TOL_FACTOR")
    put("prot.addtolq", P.ADDITIVITY_TOL_QUANTILE, pct(P.ADDITIVITY_TOL_QUANTILE), src, "ADDITIVITY_TOL_QUANTILE")
    lo, hi = min(P.ADDITIVITY_TOL_SENSITIVITY), max(P.ADDITIVITY_TOL_SENSITIVITY)
    put("prot.addsenslo", lo, f"{lo:g}", src, "min(ADDITIVITY_TOL_SENSITIVITY)")
    put("prot.addsenshi", hi, f"{hi:g}", src, "max(ADDITIVITY_TOL_SENSITIVITY)")


# ------------------------------------------------------------------ architecture, read from the models
def architecture_numbers():
    for key, spec in DATASETS.items():
        model = spec.build_model(torch.device("cpu"))
        geo = geometry_of(model)
        src = f"models/{spec.checkpoint().name} + explorer/registry.py"
        W = model.digits.routing_module.W
        K, dv, dp = W.shape[1], W.shape[3], W.shape[4]
        put(f"arch.{key}.N", geo.N, intc(geo.N), src, "geometry_of(model).N")
        put(f"arch.{key}.G", geo.G, str(geo.G), src, "geometry_of(model).G")
        put(f"arch.{key}.T", geo.T, str(geo.T), src, "geometry_of(model).T")
        kA = max(1, int(np.ceil(P.TOPK_FRACS[0] * geo.N)))         # as in run_fidelity.topk
        ks = np.unique(np.concatenate([AT.budgets(geo.N, P.CURVE_STEPS),   # as in run_fidelity.deletion_curves
                                       [max(1, int(round(b * geo.N))) for b in P.BUDGETS]]))
        last = ks[ks < geo.N][-1] / geo.N
        put(f"arch.{key}.lastbudget", last, pct(last, 1), "explorer/attribution.py + explorer/protocol.py",
            "largest partial deletion budget k < N of the deletion grid, as a share of N")
        put(f"arch.{key}.kA", kA, str(kA), src, "ceil(TOPK_FRACS[0] * N)")
        put(f"arch.{key}.img", geo.img_size, str(geo.img_size), src, "model.img_width")
        put(f"arch.{key}.cin", spec.in_channels, str(spec.in_channels), src, "conv1.in_channels")
        put(f"arch.{key}.params", sum(p.numel() for p in model.parameters()),
            intc(sum(p.numel() for p in model.parameters())), src, "sum(p.numel())")
        put(f"arch.{key}.ntest", len(spec.dataset()), intc(len(spec.dataset())), "torchvision test split",
            "len(dataset(train=False))")
        if key == "mnist":
            put("arch.K", K, str(K), src, "W.shape[1]")
            put("arch.dv", dv, str(dv), src, "W.shape[3]")
            put("arch.dp", dp, str(dp), src, "W.shape[4]")
            put("arch.iters", model.digits.routing_module.routing_iters,
                str(model.digits.routing_module.routing_iters), src, "routing_module.routing_iters")
            k1 = model.conv1.kernel_size[0]
            k2, s2 = model.primary.conv.kernel_size[0], model.primary.conv.stride[0]
            rf = k1 + (k2 - 1) * model.conv1.stride[0]
            put("arch.k1", k1, str(k1), src, "conv1.kernel_size")
            put("arch.k2", k2, str(k2), src, "primary.conv.kernel_size")
            put("arch.s2", s2, str(s2), src, "primary.conv.stride")
            put("arch.s1", model.conv1.stride[0], str(model.conv1.stride[0]), src, "conv1.stride")
            put("arch.rf", rf, str(rf), src, "k1 + (k2-1)*s1  (receptive field)")
            put("arch.rfoverlap", rf - s2 * model.conv1.stride[0], str(rf - s2 * model.conv1.stride[0]), src,
                "rf - s2*s1 (overlap of neighbouring windows)")
            put("arch.conv1ch", model.conv1.out_channels, str(model.conv1.out_channels), src, "conv1.out_channels")


# ------------------------------------------------------------------ fidelity
METHODS = list(P.METHODS)
MKEY = {"taylor": "taylor", "taylor_frozen": "frozen", "g_A": "gA", "g_C": "gC",
        "coupling": "coupling", "act_x_coupling": "actc", "activation": "activation", "random": "random"}


def fidelity_numbers(S):
    src = "results/summary.json"
    for ds in P.DATASETS:
        ref = P.REFERENCE_VARIANT[ds]
        for set_ in ("main", "misclassified"):
            run = S["runs"].get(f"{ds}/{ref}/{set_}")
            tag = "fid" if set_ == "main" else "mis"
            put(f"{tag}.{ds}.n", run["n_images"], str(run["n_images"]), src, f"runs/{ds}/{ref}/{set_}/n_images")
            if set_ == "main":
                put(f"{tag}.{ds}.ncorrect", run["n_correct"], str(run["n_correct"]), src,
                    f"runs/{ds}/{ref}/{set_}/n_correct")
            if set_ == "main":
                for v in P.VARIANTS:
                    acc = S["runs"][f"{ds}/{v}/main"]["provenance"]["test_accuracy_full"]
                    put(f"acc.{ds}.{v}", acc, pct(acc), src, f"runs/{ds}/{v}/main/provenance/test_accuracy_full",
                        "full test split")
                accs = {v: S["runs"][f"{ds}/{v}/main"]["provenance"]["test_accuracy_full"] for v in P.SEED_VARIANTS}
                assert max(accs, key=accs.get) == ref, f"{ds}: the reference run must be the most accurate"
                put(f"acc.{ds}.ref", accs[ref], pct(accs[ref]), src, f"runs/{ds}/{ref}/main/provenance/test_accuracy_full",
                    "the most accurate run")
                put(f"acc.{ds}.seedmin", min(accs.values()), pct(min(accs.values())), src, "min over the three runs")
                put(f"acc.{ds}.seedmax", max(accs.values()), pct(max(accs.values())), src, "max over the three runs")
                bc = run["taylor_p1_by_correctness"]
                for k in ("correct", "incorrect"):
                    if k in bc:
                        put(f"fid.{ds}.taylor.p1.{k}", bc[k]["mean"], pct(bc[k]["mean"]), src,
                            f"runs/{ds}/{ref}/main/taylor_p1_by_correctness/{k}/mean")
                        put(f"fid.{ds}.count.{k}", bc[k]["n"], str(bc[k]["n"]), src,
                            f"runs/{ds}/{ref}/main/taylor_p1_by_correctness/{k}/n")
            for m in METHODS:
                if m not in run["rank"]:
                    continue
                for metric in ("p1", "p5", "rho", "tau", "rho_top"):
                    d = run["rank"][m][metric]
                    base = f"{tag}.{ds}.{MKEY[m]}.{metric}"
                    path = f"runs/{ds}/{ref}/{set_}/rank/{m}/{metric}"
                    f = (lambda v: pct(v, 2)) if metric in ("p1", "p5") else (lambda v: dec(v))
                    put(base + ".mean", d["mean"], f(d["mean"]), src, path + "/mean")
                    put(base + ".lo", d["mean_ci"][0], f(d["mean_ci"][0]), src, path + "/mean_ci/0")
                    put(base + ".hi", d["mean_ci"][1], f(d["mean_ci"][1]), src, path + "/mean_ci/1")
                    put(base + ".median", d["median"], f(d["median"]), src, path + "/median")
                    put(base + ".p05", d["p05"], f(d["p05"]), src, path + "/p05")
            if set_ == "main":
                for m, d in run["magnitude"].items():
                    for k, short in (("slope_top5", "slope"), ("rel_l1_top5", "rell1")):
                        put(f"mag.{ds}.{MKEY[m]}.{short}", d[k]["median"], dec(d[k]["median"], 2), src,
                            f"runs/{ds}/{ref}/main/magnitude/{m}/{k}/median")
                ro = run["routing"]
                for k, short in (("entropy_iter", "ent"), ("entropy_top1_iter", "enttop"),
                                 ("consensus_iter", "cons"), ("vlen_iter", "vlen")):
                    for i, v in enumerate(ro[k]["mean"]):
                        put(f"route.{ds}.{short}.{i + 1}", v, dec(v, 3), src,
                            f"runs/{ds}/{ref}/main/routing/{k}/mean/{i}")
                for k, short in (("entropy_top1_iter", "enttop"), ("consensus_iter", "cons"), ("vlen_iter", "vlen")):
                    d = ro[k + "_last_change"]
                    for stat, val in (("mean", d["mean"]), ("lo", d["mean_ci"][0]), ("hi", d["mean_ci"][1])):
                        put(f"route.{ds}.{short}.last.{stat}", val, signed(val, 3), src,
                            f"runs/{ds}/{ref}/main/routing/{k}_last_change/" + ("mean" if stat == "mean" else
                                                                              f"mean_ci/{0 if stat == 'lo' else 1}"))
                for k, short in (("routing_share", "share"), ("concentration_top1", "conc")):
                    d = ro[k]
                    put(f"route.{ds}.{short}", d["median"], dec(d["median"], 2), src,
                        f"runs/{ds}/{ref}/main/routing/{k}/median")
                    put(f"route.{ds}.{short}.lo", d["median_ci"][0], dec(d["median_ci"][0], 2), src,
                        f"runs/{ds}/{ref}/main/routing/{k}/median_ci/0")
                    put(f"route.{ds}.{short}.hi", d["median_ci"][1], dec(d["median_ci"][1], 2), src,
                        f"runs/{ds}/{ref}/main/routing/{k}/median_ci/1")
                for m in ("exact", "taylor", "g_C", "g_A", "activation"):
                    d = run["deletion"][m]
                    mk = "exact" if m == "exact" else MKEY[m]
                    for k in ("del_1", "del_2", "del_5", "del_10", "min", "rebound"):
                        put(f"del.{ds}.{mk}.{k.replace('_', '')}", d[k]["median"], dec(d[k]["median"], 2), src,
                            f"runs/{ds}/{ref}/main/deletion/{m}/{k}/median")
                    put(f"del.{ds}.{mk}.argmin", d["argmin_frac"]["median"], pct(d["argmin_frac"]["median"], 1), src,
                        f"runs/{ds}/{ref}/main/deletion/{m}/argmin_frac/median")
            else:
                for i, v in enumerate(run["routing"]["entropy_top1_iter"]["mean"]):
                    put(f"mis.route.{ds}.enttop.{i + 1}", v, dec(v, 3), src,
                        f"runs/{ds}/{ref}/misclassified/routing/entropy_top1_iter/mean/{i}")
                put(f"del.{ds}.mis.exact.rebound", run["deletion"]["exact"]["rebound"]["median"],
                    dec(run["deletion"]["exact"]["rebound"]["median"], 2), src,
                    f"runs/{ds}/{ref}/misclassified/deletion/exact/rebound/median")
            # intervals for the main-vs-misclassified comparisons (one per set; the sets are disjoint)
            tag = "" if set_ == "main" else "mis."
            d = run["deletion"]["exact"]["rebound"]["median_ci"]
            for stat, i in (("lo", 0), ("hi", 1)):
                put(f"del.{ds}.{tag}exact.rebound.{stat}", d[i], dec(d[i], 2), src,
                    f"runs/{ds}/{ref}/{set_}/deletion/exact/rebound/median_ci/{i}")
            e = run["routing"]["entropy_top1_iter"]["mean_ci"][-1]
            for stat, i in (("lo", 0), ("hi", 1)):
                put(f"{tag}route.{ds}.enttop.final.{stat}", e[i], dec(e[i], 3), src,
                    f"runs/{ds}/{ref}/{set_}/routing/entropy_top1_iter/mean_ci/-1/{i}")


def additivity_numbers(S):
    """Group deletion against the sum of single effects (PROTOCOL.md, "Group deletion against additivity"). Shares
    of images with a rise or a rebound above the budget also get `.count`, for text where a
    share would round to a bound. Intervals the summary marks degenerate are not emitted."""
    src = "results/summary.json"
    sens = {f: f"tau{f:g}".replace(".", "") for f in P.ADDITIVITY_TOL_SENSITIVITY if f != 1.0}
    table = []                                   # (median, lo, hi) of every value in the I(S) table
    for ds in P.DATASETS:
        A, b, base = S["additivity"][ds], f"add.{ds}", f"additivity/{ds}"

        def stat(key, d, path, f, which="median", with_ci=True):
            if which not in d:
                return
            put(key, d[which], f(d[which]), src, f"{path}/{which}")
            ci = which + "_ci"
            method = d.get("ci_method" if which == "median" else "mean_ci_method")
            for s, i in (("lo", 0), ("hi", 1)):
                if with_ci and ci in d and method != "degenerate":
                    put(f"{key}.{s}", d[ci][i], f(d[ci][i]), src, f"{path}/{ci}/{i}")

        def count(key, d, path):
            c = d["mean"] * d["n"]
            assert abs(c - round(c)) < 1e-6, key
            put(key, int(round(c)), intc(c), src, f"{path}/mean * {path}/n")

        n = A["n_images"]
        put(f"{b}.n", n, intc(n), src, f"{base}/n_images")
        put(f"{b}.nwrong", n - A["n_correct"], intc(n - A["n_correct"]), src, f"{base}/n_images - n_correct")
        for k in ("tau", "tau_c"):
            put(f"{b}.{k.replace('_', '')}", A["calibration"][k], sci(A["calibration"][k]), src,
                f"{base}/calibration/{k}")

        tp = f"{base}/by_tau/1.0"
        T = A["by_tau"]["1.0"]
        for k, short in (("pre_rise", "rise"), ("frozen_pre_rise", "frozenrise"), ("frozen_own_pre_rise", "ownrise")):
            stat(f"{b}.{short}", T[k], f"{tp}/{k}", lambda v: dec(v, 2))
            count(f"{b}.{short}.count", T[k + "_above_tol"], f"{tp}/{k}_above_tol")
            stat(f"{b}.{short}.above", T[k + "_above_tol"], f"{tp}/{k}_above_tol", pct, "mean")
        for k, short in (("kplus_frac", "kplusfrac"), ("kminus_frac", "kminusfrac"), ("n_null_frac", "nullfrac"),
                         ("frozen_prefix_frac", "prefixfrac"), ("frozen_own_kplus_frac", "ownkplusfrac"),
                         ("pre_kmin_over_kplus", "kminrel")):
            stat(f"{b}.{short}", T[k], f"{tp}/{k}", pct)
        for f, tag in sens.items():
            Tf, pf = A["by_tau"][str(f)], f"{base}/by_tau/{f}"
            for k, short in (("pre_rise", "rise"), ("frozen_pre_rise", "frozenrise"),
                             ("frozen_own_pre_rise", "ownrise")):
                put(f"{b}.{tag}.{short}.above", Tf[k + "_above_tol"]["mean"], pct(Tf[k + "_above_tol"]["mean"]),
                    src, f"{pf}/{k}_above_tol/mean")
            put(f"{b}.{tag}.kplusfrac", Tf["kplus_frac"]["median"], pct(Tf["kplus_frac"]["median"]), src,
                f"{pf}/kplus_frac/median")

        for k, short, which, f in (("flip_at_pre_kmin", "kmin.flip", "mean", pct),
                                   ("has_k_floor", "hasfloor", "mean", pct),
                                   ("grid_rebound_above_budget", "abovebudget", "mean", pct),
                                   ("rebound_budget", "budget", "median", lambda v: dec(v, 2)),
                                   ("cos_route_at_pre_kmin", "kmin.cosroute", "median", lambda v: dec(v, 2))):
            stat(f"{b}.{short}", A[k], f"{base}/{k}", f, which)
        count(f"{b}.abovebudget.count", A["grid_rebound_above_budget"], f"{base}/grid_rebound_above_budget")

        for c, short in (("route", "route"), ("frozen", "frozen"), ("frozen_own", "frozenown")):
            for k, f in (("min", lambda v: dec(v, 3)), ("rebound", lambda v: dec(v, 2)), ("argmin_frac", pct)):
                # the location takes a few grid values only, so its bootstrap interval says nothing
                stat(f"{b}.grid.{short}.{k.split('_')[0]}", A["grid"][c][k], f"{base}/grid/{c}/{k}", f,
                     with_ci=k != "argmin_frac")

        for p, short in (("b1", "b1"), ("b2", "b2"), ("b5", "b5"), ("b10", "b10"), ("pre_kmin", "kmin"),
                         ("kplus", "kplus")):
            d, pp = A["points"][p], f"{base}/points/{p}"
            for k, sk, f in (("I_R", "IR", lambda v: dec(v, 2)), ("I_F", "IF", lambda v: dec(v, 2)),
                             ("R", "R", lambda v: signed(v, 3)), ("route_share", "share", lambda v: dec(v, 2)),
                             ("cos_frozen", "cosfrozen", lambda v: dec(v, 2))):
                stat(f"{b}.{short}.{sk}", d[k], f"{pp}/{k}", f)
                if k in ("I_R", "I_F", "R"):
                    table.append((k == "R", d[k]["median"], *d[k]["median_ci"]))
            put(f"{b}.{short}.share.n", d["route_share"]["n"], intc(d["route_share"]["n"]), src,
                f"{pp}/route_share/n")

        for c, g in A["by_correctness"].items():
            put(f"{b}.{c}.rise", g["pre_rise"]["median"], dec(g["pre_rise"]["median"], 2), src,
                f"{base}/by_correctness/{c}/pre_rise/median")
    for is_r, key, f, which in ((False, "add.table.ciwidth", lambda v: dec(v, 2), "I_R, I_F"),
                                (True, "add.table.ciwidthR", lambda v: dec(v, 3), "R")):
        w = max(max(hi - m, m - lo) for r, m, lo, hi in table if r == is_r)
        put(key, w, f(w), src, f"max |median_ci - median| of {which} over additivity/*/points/*",
            "every interval of these values in the I(S) table lies within this distance of its median")
    assert all(S["additivity"][ds]["by_tau"][str(min(P.ADDITIVITY_TOL_SENSITIVITY))]["pre_rise_above_tol"]["mean"]
               == 1.0 for ds in P.DATASETS), "the text says: every image at the smallest tolerance"
    assert S["additivity"]["mnist"]["n_correct"] == S["additivity"]["mnist"]["n_images"], \
        "the text says the MNIST additivity set holds no misclassified image"


def confirmatory_numbers(S):
    src = "results/summary.json"
    for i, r in enumerate(S["confirmatory"]):
        b = f"conf.{r['hypothesis']}.{r['dataset']}"
        pp = lambda v: f"{100 * v:.1f}"                      # percentage points of P@1%
        put(b + ".diff", r["mean_diff"], pp(r["mean_diff"]), src, f"confirmatory/{i}/mean_diff",
            "Taylor minus comparator, percentage points of P@1%")
        put(b + ".lo", r["mean_diff_ci"][0], pp(r["mean_diff_ci"][0]), src, f"confirmatory/{i}/mean_diff_ci/0",
            "percentage points")
        put(b + ".hi", r["mean_diff_ci"][1], pp(r["mean_diff_ci"][1]), src, f"confirmatory/{i}/mean_diff_ci/1",
            "percentage points")
        put(b + ".rrb", r["rank_biserial"], f"{r['rank_biserial']:+.2f}", src, f"confirmatory/{i}/rank_biserial")
        put(b + ".holm", r["holm_p"], sci(r["holm_p"]), src, f"confirmatory/{i}/holm_p")
        put(b + ".nnonzero", r["n_nonzero"], str(r["n_nonzero"]), src, f"confirmatory/{i}/n_nonzero")
        put(b + ".n", r["n"], str(r["n"]), src, f"confirmatory/{i}/n")
        put(b + ".nbetter", r["n_taylor_better"], str(r["n_taylor_better"]), src,
            f"confirmatory/{i}/n_taylor_better", "images on which Taylor has the higher P@1%")
    nrej = sum(r["reject"] for r in S["confirmatory"])
    put("conf.nreject", nrej, str(nrej), src, "sum(confirmatory/*/reject)")
    rr = [r["rank_biserial"] for r in S["confirmatory"]]
    put("conf.rrb.min", min(rr), f"{min(rr):+.2f}", src, "min(confirmatory/*/rank_biserial)")
    put("conf.rrb.max", max(rr), f"{max(rr):+.2f}", src, "max(confirmatory/*/rank_biserial)")
    worst = max(r["holm_p"] for r in S["confirmatory"])
    put("conf.maxholm", worst, sci(worst), src, "max(confirmatory/*/holm_p)")


def robustness_numbers(S):
    src = "results/summary.json"
    for ds in P.DATASETS:
        R = S["robustness"][ds]
        for m in ("taylor", "taylor_frozen", "coupling", "g_C"):
            d = R[m]["p1"]
            put(f"rob.{ds}.{MKEY[m]}.p1.min", d["min"], pct(d["min"]), src, f"robustness/{ds}/{m}/p1/min")
            put(f"rob.{ds}.{MKEY[m]}.p1.max", d["max"], pct(d["max"]), src, f"robustness/{ds}/{m}/p1/max")
        for k, short in (("routing_share", "share"), ("concentration_top1", "conc"),
                         ("exact_deletion_rebound", "rebound")):
            d = R[k]
            put(f"rob.{ds}.{short}.min", d["min"], dec(d["min"], 2), src, f"robustness/{ds}/{k}/min")
            put(f"rob.{ds}.{short}.max", d["max"], dec(d["max"], 2), src, f"robustness/{ds}/{k}/max")
            seeds = [d["per_checkpoint_median"][v] for v in P.SEED_VARIANTS]
            put(f"rob.{ds}.{short}.seedmin", min(seeds), dec(min(seeds), 2), src,
                f"min over seeds of robustness/{ds}/{k}/per_checkpoint_median")
            put(f"rob.{ds}.{short}.seedmax", max(seeds), dec(max(seeds), 2), src,
                f"max over seeds of robustness/{ds}/{k}/per_checkpoint_median")
        for m in ("taylor", "taylor_frozen", "coupling", "g_C"):
            seeds = [R[m]["p1"]["per_checkpoint_mean"][v] for v in P.SEED_VARIANTS]
            put(f"rob.{ds}.{MKEY[m]}.p1.seedmin", min(seeds), pct(min(seeds), 2), src,
                f"min over seeds of robustness/{ds}/{m}/p1/per_checkpoint_mean")
            put(f"rob.{ds}.{MKEY[m]}.p1.seedmax", max(seeds), pct(max(seeds), 2), src,
                f"max over seeds of robustness/{ds}/{m}/p1/per_checkpoint_mean")


def run_range_numbers(S):
    """Ranges over the three runs."""
    src = "results/summary.json"
    rng = lambda vals: (min(vals), max(vals))
    for ds in P.DATASETS:
        R = S["robustness"][ds]
        for m in METHODS:
            if m in ("taylor", "taylor_frozen", "coupling", "g_C") or m not in R:
                continue
            lo, hi = rng([R[m]["p1"]["per_checkpoint_mean"][v] for v in P.SEED_VARIANTS])
            put(f"rob.{ds}.{MKEY[m]}.p1.seedmin", lo, pct(lo, 2), src,
                f"min over seeds of robustness/{ds}/{m}/p1/per_checkpoint_mean")
            put(f"rob.{ds}.{MKEY[m]}.p1.seedmax", hi, pct(hi, 2), src,
                f"max over seeds of robustness/{ds}/{m}/p1/per_checkpoint_mean")
        for m in ("taylor", "taylor_frozen"):
            lo, hi = rng([R[m]["tau"]["per_checkpoint_median"][v] for v in P.SEED_VARIANTS])
            put(f"rob.{ds}.{MKEY[m]}.tau.seedrange", [lo, hi], dec(lo) if dec(lo) == dec(hi) else f"{dec(lo)}--{dec(hi)}",
                src, f"range over seeds of robustness/{ds}/{m}/tau/per_checkpoint_median")
            put(f"rob.{ds}.{MKEY[m]}.tau.seedmin", lo, dec(lo), src,
                f"min over seeds of robustness/{ds}/{m}/tau/per_checkpoint_median")
            put(f"rob.{ds}.{MKEY[m]}.tau.seedmax", hi, dec(hi), src,
                f"max over seeds of robustness/{ds}/{m}/tau/per_checkpoint_median")
        for other in ("taylor_frozen", "coupling"):
            d = [R["taylor"]["p1"]["per_checkpoint_mean"][v] - R[other]["p1"]["per_checkpoint_mean"][v]
                 for v in P.SEED_VARIANTS]
            pp = lambda v: f"{100 * v:.1f}"
            put(f"rob.{ds}.diff.{MKEY[other]}.seedmin", min(d), pp(min(d)), src,
                f"min over seeds of taylor minus {other} mean P@1%", "percentage points")
            put(f"rob.{ds}.diff.{MKEY[other]}.seedmax", max(d), pp(max(d)), src,
                f"max over seeds of taylor minus {other} mean P@1%", "percentage points")
        for m in P.MAGNITUDE_METHODS:
            for k, short, f in (("slope_top5", "slope", lambda v: dec(v, 2)), ("rel_l1_top5", "rell1", lambda v: dec(v, 2))):
                lo, hi = rng([S["runs"][f"{ds}/{v}/main"]["magnitude"][m][k]["median"] for v in P.SEED_VARIANTS])
                put(f"mag.{ds}.{MKEY[m]}.{short}.seedmin", lo, f(lo), src,
                    f"min over seeds of runs/{ds}/seed*/main/magnitude/{m}/{k}/median")
                put(f"mag.{ds}.{MKEY[m]}.{short}.seedmax", hi, f(hi), src,
                    f"max over seeds of runs/{ds}/seed*/main/magnitude/{m}/{k}/median")
        d10 = lambda v, m: S["runs"][f"{ds}/{v}/main"]["deletion"][m]["del_10"]["median"]
        more = sum(d10(v, "exact") > max(d10(v, "g_A"), d10(v, "g_C")) for v in P.SEED_VARIANTS)
        put(f"del.{ds}.rep.exactmore", more, str(more), src,
            "count of seeds where the exact ranking leaves more confidence at budget4 than both sensitivities")
        for k in ("min", "rebound"):
            lo, hi = rng([S["runs"][f"{ds}/{v}/main"]["deletion"]["exact"][k]["median"] for v in P.SEED_VARIANTS])
            put(f"del.{ds}.exact.{k}.seedmin", lo, dec(lo, 2), src,
                f"min over seeds of runs/{ds}/seed*/main/deletion/exact/{k}/median")
            put(f"del.{ds}.exact.{k}.seedmax", hi, dec(hi, 2), src,
                f"max over seeds of runs/{ds}/seed*/main/deletion/exact/{k}/median")


def margin_numbers():
    """Capsule length against the decision margin (results/summary_margin.json)."""
    src = "results/summary_margin.json"
    M = load(src)["runs"]
    stats = (("taylor_margin_p1", "p1", "mean", pct), ("length_margin_overlap", "overlap", "mean", pct),
             ("length_margin_tau", "tau", "median", lambda v: dec(v, 2)),
             ("taylor_length_vs_margin_p1", "lengthp1", "mean", pct),
             ("share_runner_up_changes", "runnerup", "mean", pct),
             ("share_prediction_changes", "flip", "mean", lambda v: pct(v, 2)))
    for ds in P.DATASETS:
        ref = P.REFERENCE_VARIANT[ds]
        refr, seeds = M[f"{ds}/{ref}"], [M[f"{ds}/{v}"] for v in P.SEED_VARIANTS]
        assert all(r["matches_stored_taylor_p1"] == 1.0 for r in seeds), "margin runs reproduce P@1%"
        for k, short, which, f in stats:
            d, path = refr[k], f"runs/{ds}/{ref}/{k}"
            put(f"mar.{ds}.{short}", d[which], f(d[which]), src, f"{path}/{which}")
            ci = which + "_ci"
            if ci in d and d.get("ci_method" if which == "median" else "mean_ci_method") != "degenerate":
                for s, i in (("lo", 0), ("hi", 1)):
                    put(f"mar.{ds}.{short}.{s}", d[ci][i], f(d[ci][i]), src, f"{path}/{ci}/{i}")
            vals = [r[k][which] for r in seeds]
            put(f"mar.{ds}.{short}.seedmin", min(vals), f(min(vals)), src, f"min over seeds of runs/{ds}/seed*/{k}/{which}")
            put(f"mar.{ds}.{short}.seedmax", max(vals), f(max(vals)), src, f"max over seeds of runs/{ds}/seed*/{k}/{which}")
        for q, short in (("0.05", "q05"), ("0.95", "q95")):
            v = refr["top_set_ratio"]["quantiles"][q]
            put(f"mag.{ds}.ratio.{short}", v, dec(v, 2), src, f"runs/{ds}/{ref}/top_set_ratio/quantiles/{q}")
            vals = [r["top_set_ratio"]["quantiles"][q] for r in seeds]
            put(f"mag.{ds}.ratio.{short}.seedmin", min(vals), dec(min(vals), 2), src,
                f"min over seeds of runs/{ds}/seed*/top_set_ratio/quantiles/{q}")
            put(f"mag.{ds}.ratio.{short}.seedmax", max(vals), dec(max(vals), 2), src,
                f"max over seeds of runs/{ds}/seed*/top_set_ratio/quantiles/{q}")
        v = refr["top_set_ratio"]["share_far"]
        put(f"mag.{ds}.far", v, pct(v), src, f"runs/{ds}/{ref}/top_set_ratio/share_far")
        vals = [r["top_set_ratio"]["share_far"] for r in seeds]
        put(f"mag.{ds}.far.seedmin", min(vals), pct(min(vals)), src, f"min over seeds of runs/{ds}/seed*/top_set_ratio/share_far")
        put(f"mag.{ds}.far.seedmax", max(vals), pct(max(vals)), src, f"max over seeds of runs/{ds}/seed*/top_set_ratio/share_far")


def sanity_numbers():
    stages = {"routing W": "W", "+ primary conv": "primary", "+ conv1": "conv1"}
    for ds in P.DATASETS:
        src = f"results/sanity/{ds}.json"
        s = load(src)["median_similarity"]
        for st, short in stages.items():
            for m in ("exact", "taylor", "taylor_frozen", "g_A", "g_C", "coupling", "activation"):
                v = s[st][m]
                put(f"san.{ds}.{short}.{MKEY.get(m, m)}", v, dec(v, 2), src, f"median_similarity/{st}/{m}")
            v = s[st]["taylor_vs_exact_randomized"]
            put(f"san.{ds}.{short}.tve", v, dec(v, 3), src, f"median_similarity/{st}/taylor_vs_exact_randomized")
        recs = load(src)["records"]
        for st, short in stages.items():
            v = float(np.mean([r["stages"][st]["pred"] != r["pred_trained"] for r in recs]))
            put(f"san.{ds}.{short}.classchange", v, pct(v, 0), src,
                f"mean(records/*/stages/{st}/pred != records/*/pred_trained)")
        aware = [m for m in P.SANITY_METHODS if m != "activation"]
        same = [r for r in recs if r["stages"]["routing W"]["pred"] == r["pred_trained"]]
        put(f"san.{ds}.W.nsame", len(same), str(len(same)), src,
            "count(records/* with stages/routing W/pred == pred_trained)")
        if same:
            v = max(abs(float(np.median([r["stages"]["routing W"]["similarity"][m] for r in same])))
                    for m in ("exact", "taylor"))
            put(f"san.{ds}.W.samemax", v, dec(v, 2), src,
                "max |median similarity| of exact and taylor, routing W stage, images whose class is unchanged")
        absmax = max(abs(s[st][m]) for st in stages for m in aware)
        put(f"san.{ds}.absmax", absmax, dec(absmax, 3), src,
            f"max |median_similarity| over stages and {aware}")


def cost_numbers():
    src = "results/cost.json"
    c = load(src)["results"]
    for ds in P.DATASETS:
        for dev in ("cuda", "cpu"):
            for k, short in (("forward", "fwd"), ("gradient_attributions", "grad"), ("exact_ablation", "exact")):
                v = c[ds][dev][k]["median_ms"]
                txt = f"{v:.1f}" if short == "grad" else ms(v)        # one decimal: the ratio is checkable
                put(f"cost.{ds}.{dev}.{short}", v, txt, src, f"results/{ds}/{dev}/{k}/median_ms")
            ratio = c[ds][dev]["exact_ablation"]["median_ms"] / c[ds][dev]["gradient_attributions"]["median_ms"]
            put(f"cost.{ds}.{dev}.ratio", ratio, f"{ratio:,.0f}".replace(",", "{,}"), src,
                f"exact_ablation/gradient_attributions ({dev})")
        app = c[ds]["cuda"]["app"]
        put(f"cost.{ds}.app.analysis", app["analysis_ms"], ms(app["analysis_ms"]), src, f"results/{ds}/cuda/app/analysis_ms")
        put(f"cost.{ds}.app.page", app["page_build_ms"], ms(app["page_build_ms"]), src, f"results/{ds}/cuda/app/page_build_ms")
        put(f"cost.{ds}.app.mb", app["page_mb"], dec(app["page_mb"], 1), src, f"results/{ds}/cuda/app/page_mb")
    prov = load(src)["provenance"]
    put("cost.gpu", prov["device"], prov["device"].replace("NVIDIA GeForce ", ""), src, "provenance/device")
    put("env.cputhreads", prov["cpu_threads"], str(prov["cpu_threads"]), src, "provenance/cpu_threads")
    for k in ("python", "torch", "cuda"):
        v = prov["versions"][k].split("+")[0]
        put(f"env.{k}", v, v, src, f"provenance/versions/{k}")


def gate_numbers():
    src = "results/gates.json"
    a = load(src)
    for ds in P.DATASETS:
        rows = a[ds]["rows"]
        md = a[ds]["max_abs_diff"]
        for k in ("p1", "tau", "rho", "phi"):
            put(f"gate.{ds}.maxdiff.{k}", md[k], sci(md[k]) if md[k] else "0", src, f"{ds}/max_abs_diff/{k}")
        put(f"gate.{ds}.predagree", a[ds]["pred_agreement"], pct(a[ds]["pred_agreement"], 0), src, f"{ds}/pred_agreement")
        f = lambda key: float(np.median([r[key] for r in rows]))
        put(f"gate.{ds}.r2", f("loglog_r2_top5"), dec(f("loglog_r2_top5"), 4), src, f"median({ds}/rows/*/loglog_r2_top5)")
        put(f"gate.{ds}.r2min", min(r["loglog_r2_top5"] for r in rows),
            dec(min(r["loglog_r2_top5"] for r in rows), 3), src, f"min({ds}/rows/*/loglog_r2_top5)")
        put(f"gate.{ds}.llslope", f("loglog_slope_top5"), dec(f("loglog_slope_top5"), 2), src,
            f"median({ds}/rows/*/loglog_slope_top5)")
        put(f"gate.{ds}.resid", f("loglog_resid_sd_top5"), dec(f("loglog_resid_sd_top5"), 3), src,
            f"median({ds}/rows/*/loglog_resid_sd_top5)")
        put(f"gate.{ds}.relerr", f("rel_err_taylor_top1pct"), pct(f("rel_err_taylor_top1pct"), 0), src,
            f"median({ds}/rows/*/rel_err_taylor_top1pct)")
        put(f"gate.{ds}.gapratio", f("cutoff_gap_ratio"), dec(f("cutoff_gap_ratio"), 3), src,
            f"median({ds}/rows/*/cutoff_gap_ratio)")
        put(f"gate.{ds}.ratiocut", f("cutoff_ratio_median"), dec(f("cutoff_ratio_median"), 2), src,
            f"median({ds}/rows/*/cutoff_ratio_median)")
        put(f"gate.{ds}.cutsd", f("cutoff_logratio_sd"), dec(f("cutoff_logratio_sd"), 3), src,
            f"median({ds}/rows/*/cutoff_logratio_sd)")
        rc = [r["cutoff_ratio_median"] for r in rows]
        put(f"gate.{ds}.ratiocut.min", min(rc), dec(min(rc), 2), src, f"min({ds}/rows/*/cutoff_ratio_median)")
        put(f"gate.{ds}.ratiocut.max", max(rc), dec(max(rc), 2), src, f"max({ds}/rows/*/cutoff_ratio_median)")
        lg = float(np.median([np.log(r["cutoff_gap_ratio"]) for r in rows]))
        put(f"gate.{ds}.loggap", lg, dec(lg, 3), src, f"median({ds}/rows/*/log cutoff_gap_ratio)")
        lr = lambda g: np.log(g["taylor"] / g["exact"])
        nb = float(np.median([abs(lr(r["gates"]["rank_k"]) - lr(r["gates"]["rank_k1"])) for r in rows]))
        put(f"gate.{ds}.nbrdiff", nb, dec(nb, 3), src,
            f"median({ds}/rows/*/|log(taylor/exact) at gates rank_k minus rank_k1|)")
        above = [abs(lr(r["gates"]["rank_k"]) - lr(r["gates"]["rank_k1"])) > np.log(r["cutoff_gap_ratio"]) for r in rows]
        put(f"gate.{ds}.nbrabove", sum(above), str(sum(above)), src,
            f"count({ds}/rows/* where the difference of log(taylor/exact) between ranks k and k+1 exceeds log gap)")
        put(f"gate.{ds}.nbrbelow", len(rows) - sum(above), str(len(rows) - sum(above)), src,
            f"count({ds}/rows/* where it does not)")
        assert all(a for a, r in zip(above, rows) if r["misses"]), "the text says misses occur only where it exceeds"
        t1k = min(r["gates"]["rank1"]["taylor"] / r["gates"]["rank_k"]["taylor"] for r in rows)
        put(f"gate.{ds}.top1taylor", t1k, dec(t1k, 1), src, f"min({ds}/rows/*/gates/rank1/taylor / gates/rank_k/taylor)")
        p1 = float(np.mean([r["p1_independent"] for r in rows]))
        put(f"gate.{ds}.p1mean", p1, pct(p1), src, f"mean({ds}/rows/*/p1_independent)")
        ok = [r["boundary_pairs_ordered"] for r in rows]
        put(f"gate.{ds}.pairsmin", min(ok), pct(min(ok)), src, f"min({ds}/rows/*/boundary_pairs_ordered)")
        top = [r["ratio_top1"] for r in rows]
        put(f"gate.{ds}.ratiotop.min", min(top), dec(min(top), 2), src, f"min({ds}/rows/*/ratio_top1)")
        put(f"gate.{ds}.ratiotop.max", max(top), dec(max(top), 2), src, f"max({ds}/rows/*/ratio_top1)")
        misses = [m for r in rows for m in r["misses"]]
        put(f"gate.{ds}.nmiss", len(misses), str(len(misses)), src, f"count({ds}/rows/*/misses)")
        if misses:
            gap = max(m["gap_to_next"] for m in misses)
            put(f"gate.{ds}.missgap", gap, dec(gap, 4), src, f"max({ds}/rows/*/misses/*/gap_to_next)")
            assert all(m["true_rank"] == int(np.ceil(P.TOPK_FRACS[0] * rows[0]["N"])) for m in misses), \
                "the text says every missed capsule is the one at the cutoff"
        fd = max(abs(g["slope_at_1_fd"] - g["taylor"]) / abs(g["taylor"]) for r in rows for g in r["gates"].values())
        put(f"gate.{ds}.gateid", fd, sci(fd), src, f"max over {ds}/rows/*/gates of |slope_at_1_fd - taylor| / |taylor|")
        put(f"gate.{ds}.uniquemin", min(r["unique_exact_values"] for r in rows),
            intc(min(r["unique_exact_values"] for r in rows)), src, f"min({ds}/rows/*/unique_exact_values)")
        for s in P.GATE_NOISE_SIGMAS:
            v = float(np.mean([r["noise_p1"][str(s)] for r in rows]))
            put(f"gate.{ds}.noise{str(s).replace('.', 'p')}", v, pct(v), src, f"mean({ds}/rows/*/noise_p1/{s})")
    assert all(r["boundary_pairs_ordered"] == 1 and not r["misses"] for r in a["mnist"]["rows"]), \
        "the text says: every pair ordered and no miss on MNIST"
    for s in P.GATE_NOISE_SIGMAS:
        put(f"gate.sigma{str(s).replace('.', 'p')}", s, f"{s:g}", "explorer/protocol.py", "GATE_NOISE_SIGMAS")
    put("prot.gatesteps", P.GATE_STEPS, str(P.GATE_STEPS), "explorer/protocol.py", "GATE_STEPS")


def case_numbers():
    for ds in P.DATASETS:
        src = f"results/cache/figdata_{ds}.npz"
        cm = json.loads(str(np.load(ROOT / src)["case_metrics"]))
        for name, m in cm.items():
            b = f"case.{ds}.{name}"
            put(b + ".p1", m["p1"], pct(m["p1"], 0), src, f"case_metrics/{name}/p1")
            put(b + ".phi", m["phi"], dec(m["phi"], 2), src, f"case_metrics/{name}/phi")
            put(b + ".top1ratio", m["top1ratio"], dec(m["top1ratio"], 2), src, f"case_metrics/{name}/top1ratio",
                "Taylor / exact for the capsule with the largest exact effect")
            put(b + ".top1exact", m["top1_exact"], dec(m["top1_exact"], 3), src, f"case_metrics/{name}/top1_exact")
            cls = DATASETS[ds].classes
            put(b + ".label", m["label"], cls[m["label"]], src, f"case_metrics/{name}/label", "class name")
            put(b + ".pred", m["pred"], cls[m["pred"]], src, f"case_metrics/{name}/pred", "class name")


def figure_numbers():
    """Pairs the log-log magnitude figure cannot draw (non-positive estimate or effect)."""
    drop = {"taylor": {}, "taylor_frozen": {}}
    for ds in P.DATASETS:
        src = f"results/cache/figdata_{ds}.npz"
        d = np.load(ROOT / src)
        ex = d["mag_exact"]
        for m in drop:
            drop[m][ds] = int(((ex <= 0) | (d[f"mag_{m}"] <= 0)).sum())
    for m, short in (("taylor", "taylor"), ("taylor_frozen", "frozen")):
        n = sum(drop[m].values())
        where = sorted(ds for ds, v in drop[m].items() if v)
        put(f"fig.magdrop.{short}", n, intc(n), "results/cache/figdata_*.npz", f"mag_exact, mag_{m}",
            "pairs omitted from the log-log plot; datasets: " + ", ".join(where))
    assert all(v == 0 for ds, v in drop["taylor_frozen"].items() if ds != "cifar10") and \
        all(v == 0 for ds, v in drop["taylor"].items() if ds != "cifar10"), "caption says: only on CIFAR-10"


def write():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by scripts/make_numbers.py -- do not edit.",
             "% \\V{key} prints a number; an unknown key stops the build.",
             "\\makeatletter",
             "\\newcommand{\\V}[1]{\\ifcsname num@#1\\endcsname\\csname num@#1\\endcsname"
             "\\else\\PackageError{numbers}{Unknown number key '#1'}{Run scripts/make_numbers.py}\\fi}",
             ]
    for k in sorted(NUMBERS):
        t = NUMBERS[k]["text"]
        body = t if NUMBERS[k].get("note") == "class name" or k == "cost.gpu" else f"\\ensuremath{{{t}}}"
        lines.append(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{body}}}")
    lines.append("\\makeatother")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(dict(sorted(NUMBERS.items())), indent=1))
    print(f"wrote {len(NUMBERS)} numbers to paper/generated/numbers.tex/.json")


def main():
    S = json.loads((RES / "summary.json").read_text())
    protocol_numbers()
    architecture_numbers()
    fidelity_numbers(S)
    additivity_numbers(S)
    confirmatory_numbers(S)
    robustness_numbers(S)
    run_range_numbers(S)
    margin_numbers()
    sanity_numbers()
    cost_numbers()
    gate_numbers()
    case_numbers()
    figure_numbers()
    write()


if __name__ == "__main__":
    main()
