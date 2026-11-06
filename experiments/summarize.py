#!/usr/bin/env python3
"""
Statistics for the fidelity runs, as specified in PROTOCOL.md.

Unit of analysis: the image, within one checkpoint. Images are never pooled across
checkpoints; robustness is reported as the range of per-checkpoint summaries.

    python experiments/summarize.py            ->  results/summary.json, results/summary_margin.json
    python experiments/summarize.py --margin   ->  results/summary_margin.json only
"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import norm, skew, wilcoxon

from common import RESULTS, md5, provenance                                  # noqa: I100

from explorer import protocol as P                                           # noqa: E402

SEED = P.SEED
DATASETS = P.DATASETS
VARIANTS = P.VARIANTS
RANK_METRICS = ["p1", "p5", "rho", "tau", "rho_top"]
DEL_METRICS = [f"del_{int(round(b * 100))}" for b in P.BUDGETS] + ["min", "argmin_frac", "rebound", "auc_log"]
CONFIRMATORY = list(P.CONFIRMATORY)
N_BOOT = P.N_BOOT


# ------------------------------------------------------------------ bootstrap
def stratified_boot(values, strata, stat, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    values, strata = np.asarray(values, float), np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        out[b] = stat(values[idx])
    return out


def ci(values, strata, stat=np.median, level=P.CI_LEVEL):
    """Class-stratified bootstrap CI: percentile, or BCa when |skewness| > P.BCA_SKEW_THRESHOLD
    (the acceleration is estimated by an unstratified leave-one-out jackknife).
    Degenerate bootstrap distributions (e.g. a median pinned at 1.0) return [θ, θ]."""
    values = np.asarray(values, float)
    theta = float(stat(values))
    boot = stratified_boot(values, strata, stat)
    a_lo, a_hi = (1 - level) / 2, 1 - (1 - level) / 2
    if np.ptp(boot) == 0:
        return theta, [theta, theta], "degenerate"
    sk = float(skew(values)) if np.ptp(values) > 0 else 0.0
    if abs(sk) <= P.BCA_SKEW_THRESHOLD:
        return theta, [float(np.quantile(boot, a_lo)), float(np.quantile(boot, a_hi))], "percentile"
    prop = np.clip(np.mean(boot < theta), 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop)
    jack = np.array([stat(np.delete(values, i)) for i in range(len(values))])
    d = jack.mean() - jack
    acc = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5) + 1e-12)
    adj = lambda a: norm.cdf(z0 + (z0 + norm.ppf(a)) / (1 - acc * (z0 + norm.ppf(a))))
    return theta, [float(np.quantile(boot, adj(a_lo))), float(np.quantile(boot, adj(a_hi)))], "BCa"


def describe(values, strata, with_ci=True):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    d = {"n": int(len(v)), "median": float(np.median(v)), "mean": float(np.mean(v)),
         "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75)),
         "p05": float(np.quantile(v, 0.05))}
    if with_ci:
        _, d["median_ci"], d["ci_method"] = ci(values, strata)
        # For bounded, discrete metrics at the ceiling (e.g. P@1% with k = ceil(0.01 N)) the median CI
        # collapses; the mean and its CI then carry the information.
        _, d["mean_ci"], d["mean_ci_method"] = ci(values, strata, stat=np.mean)
    return d


# ------------------------------------------------------------------ paired tests
def rank_biserial(diff):
    """Matched-pairs rank-biserial correlation, consistent with Wilcoxon's Pratt
    convention used for the p-value: |d| is ranked INCLUDING zero differences
    (average ranks for ties), zeros contribute to neither side, and the result is
    (R+ - R-) / (R+ + R- + R0), in [-1, 1]."""
    from scipy.stats import rankdata
    d = np.asarray(diff, float)
    if not np.any(d):
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * pvals[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------ main
def load(dataset, variant, set_):
    p = RESULTS / "fidelity" / f"{dataset}_{variant}_{set_}.json"
    return json.loads(p.read_text()) if p.exists() else None


def per_run(run):
    recs = run["records"]
    strata = [r["label"] for r in recs]
    methods = list(recs[0]["rank"])
    out = {"n_images": len(recs), "n_correct": int(sum(r["correct"] for r in recs)), "rank": {},
           "magnitude": {}, "deletion": {}, "routing": {}}
    for m in methods:
        out["rank"][m] = {k: describe([r["rank"][m][k] for r in recs], strata) for k in RANK_METRICS}
    for m in recs[0]["magnitude"]:
        out["magnitude"][m] = {k: describe([r["magnitude"][m][k] for r in recs], strata)
                               for k in ("rel_l1_top5", "slope_top5")}
    for m in recs[0]["deletion"]:
        out["deletion"][m] = {k: describe([r["deletion"][m][k] for r in recs], strata)
                              for k in DEL_METRICS}
    for k in ("entropy_iter", "entropy_top1_iter", "consensus_iter", "vlen_iter"):
        arr = np.array([r[k] for r in recs])
        out["routing"][k] = {"mean": arr.mean(0).tolist(), "median": np.median(arr, 0).tolist(),
                             "q25": np.quantile(arr, 0.25, 0).tolist(),
                             "q75": np.quantile(arr, 0.75, 0).tolist(),
                             "mean_ci": [ci(arr[:, i], strata, stat=np.mean)[1] for i in range(arr.shape[1])]}
        # change from the second-to-last to the last iteration, per image
        dlast = arr[:, -1] - arr[:, -2]
        out["routing"][k + "_last_change"] = describe(dlast, strata)
    for k in ("routing_share", "concentration_top1", "phi"):
        out["routing"][k] = describe([r[k] for r in recs], strata)
    out["p1_by_class"] = {}
    for m in methods:
        by_class = defaultdict(list)
        for r in recs:
            by_class[r["label"]].append(r["rank"][m]["p1"])
        out["p1_by_class"][m] = {int(c): {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
                                 for c, v in sorted(by_class.items())}
    by_correct = defaultdict(list)
    for r in recs:
        by_correct["correct" if r["correct"] else "incorrect"].append(r["rank"]["taylor"]["p1"])
    out["taylor_p1_by_correctness"] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in by_correct.items()}
    return out


def confirmatory(runs):
    rows = []
    for ds in DATASETS:
        ref = P.REFERENCE_VARIANT[ds]
        run = runs.get((ds, ref, "main"))
        if run is None:
            continue
        recs = run["records"]
        strata = np.array([r["label"] for r in recs])
        for h, other in CONFIRMATORY:
            if other not in recs[0]["rank"]:
                raise KeyError(f"{other} missing in {ds}/{ref}/main: the confirmatory family must be complete")
            d = np.array([r["rank"]["taylor"][P.PRIMARY_METRIC] - r["rank"][other][P.PRIMARY_METRIC]
                          for r in recs])
            mean, mci, mth = ci(d, strata, stat=np.mean)
            try:
                p = float(wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)
            except ValueError:            # all differences zero
                p = 1.0
            rows.append({"hypothesis": h, "dataset": ds, "comparison": f"taylor vs {other}",
                         "metric": P.PRIMARY_METRIC, "n": len(d), "mean_diff": mean, "mean_diff_ci": mci,
                         "ci_method": mth, "n_nonzero": int((d != 0).sum()), "n_taylor_better": int((d > 0).sum()),
                         "wilcoxon_p": p, "rank_biserial": rank_biserial(d)})
    adj = holm(np.array([r["wilcoxon_p"] for r in rows]))
    for r, a in zip(rows, adj):
        r["holm_p"] = float(a)
        r["reject"] = bool(a < P.ALPHA)
    return rows


def robustness(runs):
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        avail = [v for v in VARIANTS if (ds, v, "main") in runs]
        if not avail:
            continue
        methods = list(runs[(ds, avail[0], "main")]["records"][0]["rank"]) if avail else []
        for m in methods:
            for k in ("p1", "rho", "tau"):
                ok = [v for v in avail if m in runs[(ds, v, "main")]["records"][0]["rank"]]
                if not ok:
                    continue
                per = {v: [r["rank"][m][k] for r in runs[(ds, v, "main")]["records"]] for v in ok}
                means = {v: float(np.mean(x)) for v, x in per.items()}
                # P@1% is discrete and sits at the ceiling, so per-checkpoint MEANS carry the
                # spread; medians are kept for completeness.
                out[ds].setdefault(m, {})[k] = {
                    "per_checkpoint_mean": means, "min": min(means.values()), "max": max(means.values()),
                    "per_checkpoint_median": {v: float(np.median(x)) for v, x in per.items()}}
        for k in ("routing_share", "concentration_top1"):
            vals = {v: float(np.median([r[k] for r in runs[(ds, v, "main")]["records"]])) for v in avail}
            out[ds][k] = {"per_checkpoint_median": vals, "min": min(vals.values()), "max": max(vals.values())}
        reb = {v: float(np.median([r["deletion"]["exact"]["rebound"] for r in runs[(ds, v, "main")]["records"]]))
               for v in avail}
        out[ds]["exact_deletion_rebound"] = {"per_checkpoint_median": reb, "min": min(reb.values()),
                                             "max": max(reb.values())}
    return out


def describe_finite(values, strata, with_ci=True):
    v, s = np.asarray(values, float), np.asarray(strata)
    ok = np.isfinite(v)
    return describe(v[ok], s[ok], with_ci) if ok.any() else {"n": 0}


def additivity_run(run):
    """Group deletion against the sum of single effects (PROTOCOL.md, "Group deletion against additivity"). The
    sensitivity tolerances (factors other than 1) are summarized without CIs. A share of
    images leaves out images whose value is undefined, like the value itself."""
    if run["provenance"].get("limit"):
        raise ValueError("additivity run made with --limit (debug run); not summarized")
    recs = run["records"]
    strata = [r["label"] for r in recs]
    tau_c = run["calibration"]["tau_c"]
    get = lambda f: [f(r) for r in recs]
    above = lambda vals, tol: [float(v > tol) if np.isfinite(v) else np.nan for v in vals]
    grid_rebound = get(lambda r: r["grid"]["route"]["rebound"])
    budget = get(lambda r: r["rebound_budget"])
    out = {"n_images": len(recs), "n_correct": int(sum(r["correct"] for r in recs)),
           "calibration": run["calibration"],
           "n_zero_effects": int(sum(r["n_zero"] for r in recs)),
           "max_abs_phi0_route_minus_frozen": float(max(abs(r["phi0"] - r["phi0_frozen"]) for r in recs)),
           "flip_at_pre_kmin": describe(get(lambda r: float(r["flip_at_pre_kmin"])), strata),
           "cos_route_at_pre_kmin": describe_finite(get(lambda r: r["cos_route_at_pre_kmin"]), strata),
           "has_k_floor": describe(get(lambda r: float(r["k_floor"] is not None)), strata),
           "rebound_budget": describe(budget, strata),
           "grid_rebound_above_budget": describe_finite(
               [float(g > b + tau_c) for g, b in zip(grid_rebound, budget)], strata),
           "grid": {c: {k: describe(get(lambda r: r["grid"][c][k]), strata) for k in ("min", "argmin_frac", "rebound")}
                    for c in ("route", "frozen", "frozen_own")},
           "points": {p: {k: describe_finite(get(lambda r: r["points"][p][k]), strata)
                          for k in ("k", "I_R", "I_F", "R", "route_share", "cos_frozen")}
                      for p in recs[0]["points"]},
           "by_tau": {}}
    for f in P.ADDITIVITY_TOL_SENSITIVITY:
        t = lambda r: r["by_tau"][str(f)]
        w = f == 1.0
        frac = lambda key: describe_finite(
            get(lambda r: (np.nan if t(r)[key] is None else t(r)[key]) / r["N"]), strata, w)
        rise = {k: get(lambda r: t(r)[k]) for k in ("pre_rise", "frozen_pre_rise", "frozen_own_pre_rise")}
        out["by_tau"][str(f)] = {
            "tau_c": f * tau_c,
            **{f"{k}_frac": frac(k) for k in ("kplus", "kminus", "n_null", "pre_kmin", "frozen_prefix",
                                              "frozen_own_kplus")},
            "pre_kmin_over_kplus": describe_finite(get(lambda r: t(r)["pre_kmin"] / t(r)["kplus"]
                                                       if t(r)["kplus"] else np.nan), strata, w),
            **{k: describe_finite(v, strata, w) for k, v in rise.items()},
            **{f"{k}_above_tol": describe_finite(above(v, f * tau_c), strata, w) for k, v in rise.items()}}
    by_correct = defaultdict(lambda: defaultdict(list))
    for r in recs:
        g = by_correct["correct" if r["correct"] else "incorrect"]
        g["pre_rise"].append(r["by_tau"]["1.0"]["pre_rise"])
        g["grid_rebound"].append(r["grid"]["route"]["rebound"])
    out["by_correctness"] = {c: {k: {"median": float(np.nanmedian(v)), "n": int(np.isfinite(v).sum())}
                                 for k, v in g.items()}
                             for c, g in by_correct.items()}
    return out


def margin_run(run):
    """Length against margin (experiments/run_margin.py), one checkpoint."""
    recs = run["records"]
    strata = [r["label"] for r in recs]
    get = lambda f: [f(r) for r in recs]
    ex = np.concatenate([r["top_set_pairs"]["exact"] for r in recs])
    ta = np.concatenate([r["top_set_pairs"]["taylor"] for r in recs])
    ratio = np.where(ex > 0, ta / np.where(ex > 0, ex, 1), np.nan)
    far = (ex <= 0) | (ta <= 0) | (ratio > P.FAR_FACTOR) | (ratio < 1 / P.FAR_FACTOR)
    return {"n_images": len(recs),
            "taylor_margin_p1": describe(get(lambda r: r["taylor_margin_vs_exact_margin"]["p1"]), strata),
            "taylor_margin_tau": describe(get(lambda r: r["taylor_margin_vs_exact_margin"]["tau"]), strata),
            "length_margin_overlap": describe(get(lambda r: r["exact_length_vs_exact_margin"]["overlap_top1"]),
                                              strata),
            "length_margin_tau": describe(get(lambda r: r["exact_length_vs_exact_margin"]["tau"]), strata),
            "taylor_length_vs_margin_p1": describe(get(lambda r: r["taylor_length_vs_exact_margin"]["p1"]), strata),
            "share_runner_up_changes": describe(get(lambda r: r["share_runner_up_changes"]), strata),
            "share_prediction_changes": describe(get(lambda r: r["share_prediction_changes"]), strata),
            "matches_stored_taylor_p1": float(np.mean(get(lambda r: r["matches_stored_taylor_p1"]))),
            "top_set_ratio": {"n_pairs": int(len(ex)), "share_far": float(far.mean()),
                              "quantiles": {str(q): float(np.nanquantile(ratio, q))
                                            for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}}


def margin_summary():
    """results/summary_margin.json from results/margin/*.json."""
    out, inputs = {}, {}
    for ds in DATASETS:
        for v in VARIANTS:
            p = RESULTS / "margin" / f"{ds}_{v}_main.json"
            if p.exists():
                run = json.loads(p.read_text())
                if run["provenance"].get("limit"):
                    raise ValueError(f"{p.name} was made with --limit (debug run); not summarized")
                out[f"{ds}/{v}"] = margin_run(run)
                inputs[f"{ds}/{v}"] = md5(p)
    if out:
        dest = RESULTS / "summary_margin.json"
        dest.write_text(json.dumps({"runs": out, "provenance": provenance(inputs=inputs)}, indent=1))
        print(f"wrote {dest}  ({len(out)} runs)")


def main():
    if "--margin" in sys.argv[1:]:
        margin_summary()
        return
    runs = {}
    for ds in DATASETS:
        for v in VARIANTS:
            for s in ("main", "misclassified"):
                r = load(ds, v, s)
                if r is not None:
                    runs[(ds, v, s)] = r
    summary = {"runs": {f"{d}/{v}/{s}": {"provenance": {k: r["provenance"][k] for k in
                                                         ("checkpoint", "checkpoint_md5", "n_images",
                                                          "test_accuracy_full", "timestamp")},
                                          **per_run(r)}
                        for (d, v, s), r in runs.items()},
               "confirmatory": confirmatory(runs),
               "robustness": robustness(runs),
               "settings": {"n_boot": N_BOOT, "seed": SEED, "ci_level": P.CI_LEVEL,
                            "ci": f"class-stratified bootstrap; percentile, BCa if |skew| > {P.BCA_SKEW_THRESHOLD}",
                            "alpha": P.ALPHA},
               "provenance": provenance(inputs={f"{d}/{v}/{s}": md5(RESULTS / "fidelity" / f"{d}_{v}_{s}.json")
                                                for (d, v, s) in runs})}
    additivity = {}
    for ds in DATASETS:
        p = RESULTS / "additivity" / f"{ds}_{P.REFERENCE_VARIANT[ds]}_additivity.json"
        if p.exists():
            additivity[ds] = additivity_run(json.loads(p.read_text()))
            summary["provenance"]["inputs"][f"additivity/{ds}"] = md5(p)
    if additivity:
        summary["additivity"] = additivity
    summary["settings"]["family"] = f"{len(summary['confirmatory'])} tests, Holm"
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}  ({len(runs)} runs)")
    for r in summary["confirmatory"]:
        print(f"  {r['hypothesis']} {r['dataset']:13s} {r['comparison']:24s} Δ̄P@1% = {r['mean_diff']:+.3f} "
              f"[{r['mean_diff_ci'][0]:+.3f}, {r['mean_diff_ci'][1]:+.3f}]  r_rb = {r['rank_biserial']:+.2f}  "
              f"Holm p = {r['holm_p']:.2g}")
    margin_summary()


if __name__ == "__main__":
    main()
