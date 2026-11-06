#!/usr/bin/env python3
"""
Group deletion against the sum of single-capsule effects (PROTOCOL.md, "Group deletion against additivity").

Capsules are removed in descending order of the exact effect Δ. For every k the
confidence is recomputed with routing re-run (φ_R) and with the final couplings of
the intact input held fixed (φ_F), and compared with the additive prediction
φ0 − Σ_{i∈S_k} Δ_i through the interaction I(S) = [φ0 − φ(S)] − Σ_{i∈S} Δ_i.

    python experiments/run_additivity.py --dataset mnist
    python experiments/run_additivity.py --dataset mnist --set main --limit 3 --out /tmp/x

Writes <out>/<dataset>_<variant>_<set>.json (one record per image) and
<out>/<dataset>_<variant>_<set>_curves.npz (curves on the protocol grid).
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (RESULTS, additivity_set, main_set, md5, misclassified_set,  # noqa: I100
                    pilot_indices, provenance, set_determinism)

from run_fidelity import curve_summary                                    # noqa: E402

from explorer import attribution as AT                                    # noqa: E402
from explorer import protocol as P                                        # noqa: E402
from explorer.registry import DATASETS                                    # noqa: E402

CPU = torch.device("cpu")


def routed_deletion(model, u, j, order, ks=None, chunk=96):
    """φ after removing the first k capsules of `order` with routing re-run, for k in ks
    (default every k = 0..N; k = 0 gives φ0 from the same batches)."""
    N = u.shape[1]
    ks = np.arange(N + 1) if ks is None else np.asarray(ks)
    rank = np.empty(N, int)
    rank[order] = np.arange(N)
    keep = torch.as_tensor(rank[None, :] >= ks[:, None], device=u.device)
    return AT.phi_of_masks(model, u, keep, j, chunk=chunk).double().cpu().numpy()


def deletion_grid(N):
    """The grid of the fidelity runs: log-spaced counts plus the budgets (run_fidelity.deletion_curves)."""
    return np.unique(np.concatenate([AT.budgets(N, P.CURVE_STEPS),
                                     [max(1, int(round(b * N))) for b in P.BUDGETS]]))


def rebound(y, last):
    """Minimum of y over k = 1..last, its location, and the largest rise after it within k ≤ last."""
    if last < 1:
        return float("nan"), 0, float("nan")
    seg = y[1:last + 1]
    i = int(np.argmin(seg))
    return float(seg[i]), i + 1, float(seg[i:].max() - seg[i])


def leading_run(mask):
    """Length of the leading run of True values."""
    return int(np.argmin(mask)) if not mask.all() else len(mask)


def image_core(model, u, j):
    """Exact effects, deletion order and the routed deletion curve for one image."""
    ex = AT.exact_ablation(model, u, j).astype(np.float64)
    order = np.argsort(-ex, kind="stable")
    return ex, order, routed_deletion(model, u, j, order)


def calibrate(model, ds, idx, device):
    """float32 (on `device`) against float64 (on the CPU) on the calibration images:
    |Δ32 − Δ64| over all capsules, and |φ_R32 − φ_R64| / φ0 on the deletion grid in the
    float32 order."""
    m64 = copy.deepcopy(model).to(CPU).double()
    d_err, c_err = [], []
    for i in idx:
        x = ds[i][0].unsqueeze(0)
        u32 = AT.primary_capsules(model, x.to(device))
        j = AT.predicted_class(model, u32)
        ex32 = AT.exact_ablation(model, u32, j).astype(np.float64)
        order = np.argsort(-ex32, kind="stable")
        ks = deletion_grid(len(ex32))
        ks = ks[ks < len(ex32)]
        phi32 = routed_deletion(model, u32, j, order, ks)
        u64 = AT.primary_capsules(m64, x.double())
        if AT.predicted_class(m64, u64) != j:
            raise RuntimeError(f"float32 and float64 predictions differ on calibration image {i}")
        ex64 = AT.exact_ablation(m64, u64, j)
        phi64 = routed_deletion(m64, u64, j, order, ks)
        d_err.append(np.abs(ex32 - ex64))
        c_err.append(np.abs(phi32 - phi64)[1:] / phi64[0])
    d_err, c_err = np.concatenate(d_err), np.concatenate(c_err)
    q, f = P.ADDITIVITY_TOL_QUANTILE, P.ADDITIVITY_TOL_FACTOR
    return {"indices": [int(i) for i in idx], "quantile": q, "factor": f,
            "delta_abs_err_quantile": float(np.quantile(d_err, q)), "delta_abs_err_max": float(d_err.max()),
            "curve_rel_err_quantile": float(np.quantile(c_err, q)), "curve_rel_err_max": float(c_err.max()),
            "tau": f * float(np.quantile(d_err, q)), "tau_c": f * float(np.quantile(c_err, q))}


def drawup(y, last):
    """Largest rise of y over any earlier point within k = 1..last. An additive curve cannot
    rise while only capsules with positive effects are being removed."""
    if last < 1:
        return float("nan")
    seg = y[1:last + 1]
    return float((seg - np.minimum.accumulate(seg)).max())


def thresholded(ex, exf, order, y_r, y_f, y_ff, tau):
    """Every quantity that depends on the effect tolerance τ."""
    N = len(ex)
    kplus = int((ex > tau).sum())
    n_neg = int((ex < -tau).sum())
    last = min(kplus, N - 1)
    pre = rebound(y_r, last)
    prefix_f = leading_run(exf[order] > tau)
    kplus_own = int((exf > tau).sum())
    return {"kplus": kplus, "kminus": N - n_neg + 1 if n_neg else None, "n_null": N - kplus - n_neg,
            "pre_min": pre[0], "pre_kmin": pre[1], "pre_rise": drawup(y_r, last),
            "frozen_prefix": prefix_f, "frozen_pre_rise": drawup(y_f, min(prefix_f, N - 1)),
            "frozen_own_kplus": kplus_own, "frozen_own_pre_rise": drawup(y_ff, min(kplus_own, N - 1))}


def analyse(model, u, j, ex, order, phi_r, tau):
    N = u.shape[1]
    phi0 = phi_r[0]
    w = AT.frozen_votes(model, u, j)
    exf = AT.frozen_ablation(w)
    psi_f, phi_f, cos_f = AT.frozen_deletion(w, order)
    phi_ff = AT.frozen_deletion(w, np.argsort(-exf, kind="stable"))[1]

    add = np.concatenate([[0.0], np.cumsum(ex[order])])                # Σ_{S_k} Δ_i, k = 0..N
    add_f = np.concatenate([[0.0], np.cumsum(exf[order])])
    i_r = (phi0 - phi_r) - add
    i_f = (phi_f[0] - phi_f) - add_f
    y_r, y_f, y_ff = phi_r / phi0, phi_f / phi_f[0], phi_ff / phi_ff[0]

    by_tau = {str(f): thresholded(ex, exf, order, y_r, y_f, y_ff, f * tau)
              for f in P.ADDITIVITY_TOL_SENSITIVITY}
    kmin = max(1, by_tau["1.0"]["pre_kmin"])
    with torch.no_grad():
        v0 = model.digits(u)[0][0, j]
        keep = torch.ones(1, N, device=u.device, dtype=u.dtype)
        keep[0, torch.as_tensor(order[:kmin].copy(), device=u.device)] = 0
        v, *_ = model.digits(u * keep.unsqueeze(-1))
    cos_r = float(torch.dot(v[0, j], v0) / (v[0, j].norm() * v0.norm()).clamp_min(1e-30))
    floor = np.flatnonzero(add >= phi0)

    grid = deletion_grid(N)
    points = {f"b{int(round(b * 100))}": max(1, int(round(b * N))) for b in P.BUDGETS}
    points["pre_kmin"] = kmin
    points["kplus"] = max(1, min(by_tau["1.0"]["kplus"], N - 1))

    def at(k):
        # The routing share is left out where I_R − I_F is within the float32 error that
        # the k summed single effects can carry.
        ir, if_ = i_r[k] / phi0, i_f[k] / phi0
        den = abs(if_) + abs(ir - if_)
        share = abs(ir - if_) / den if abs(ir - if_) > k * tau / phi0 else float("nan")
        return {"k": int(k), "I_R": float(ir), "I_F": float(if_), "R": float(y_r[k] - phi_f[k] / phi0),
                "route_share": float(share), "cos_frozen": float(cos_f[k])}

    rec = {"phi0": float(phi0), "phi0_frozen": float(phi_f[0]), "N": N,
           "n_pos": int((ex > 0).sum()), "n_neg": int((ex < 0).sum()), "n_zero": int((ex == 0).sum()),
           "rebound_budget": float(-ex[ex < 0].sum() / phi0),
           "flip_at_pre_kmin": int(v[0].norm(dim=-1).argmax()) != j, "cos_route_at_pre_kmin": cos_r,
           "k_floor": int(floor[0]) if len(floor) else None,
           "grid": {"route": curve_summary(grid, y_r[grid], N), "frozen": curve_summary(grid, y_f[grid], N),
                    "frozen_own": curve_summary(grid, y_ff[grid], N)},
           "by_tau": by_tau, "points": {name: at(k) for name, k in points.items()}}
    curves = {"route": y_r, "frozen": y_f, "additive": 1 - add / phi0, "I_R": i_r / phi0,
              "I_F": i_f / phi0, "frozen_norm_s": psi_f / psi_f[0], "frozen_cos_s": cos_f}
    return rec, {k: c[grid] for k, c in curves.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--set", default="additivity", choices=["additivity", "main"])
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: first n images only, of the image set and of the calibration set")
    ap.add_argument("--out", type=Path, default=RESULTS / "additivity")
    args = ap.parse_args()
    set_determinism()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec, variant = DATASETS[args.dataset], P.REFERENCE_VARIANT[args.dataset]
    model = spec.build_model(device, variant)
    ds = spec.dataset(train=False)
    targets = spec.targets(ds)

    main_idx = main_set(targets)
    pilot = pilot_indices(len(targets))
    # The misclassified set that was analysed is the recorded one; recomputing it can differ
    # across devices when a borderline prediction changes the seeded draw.
    recorded = RESULTS / "fidelity" / f"{args.dataset}_{variant}_misclassified.json"
    if recorded.exists():
        rec_prov = json.loads(recorded.read_text())["provenance"]
        mis_idx, acc = rec_prov["indices"], rec_prov["test_accuracy_full"]
    else:
        mis_idx, acc = misclassified_set(model, ds, targets, device, exclude=set(main_idx) | pilot)
    if args.set == "additivity":
        idx = additivity_set(len(targets), set(main_idx) | pilot | set(mis_idx))
    else:
        idx = main_idx
    cal_idx = [i for c in range(len(spec.classes))
               for i in main_idx[c * P.N_PER_CLASS: c * P.N_PER_CLASS + P.ADDITIVITY_CALIBRATION_PER_CLASS]]
    if args.limit:
        idx, cal_idx = idx[:args.limit], cal_idx[:args.limit]

    t0 = time.time()
    cal = calibrate(model, ds, cal_idx, device)
    print(f"  calibration: τ = {cal['tau']:.3g}, τ_c = {cal['tau_c']:.3g}  ({time.time() - t0:.0f}s)", flush=True)

    records, curves, t0 = [], [], time.time()
    N = None
    for n, i in enumerate(idx):
        x = ds[i][0].unsqueeze(0).to(device)
        u = AT.primary_capsules(model, x)
        j = AT.predicted_class(model, u)
        ex, order, phi_r = image_core(model, u, j)
        rec, cur = analyse(model, u, j, ex, order, phi_r, cal["tau"])
        records.append({"idx": int(i), "label": int(targets[i]), "pred": j, "correct": bool(j == targets[i]),
                        **rec})
        N = rec["N"]
        curves.append(np.stack([cur[k] for k in sorted(cur)]))
        if (n + 1) % 100 == 0 or n + 1 == len(idx):
            print(f"  {n + 1}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{args.dataset}_{variant}_{args.set}"
    np.savez_compressed(f"{stem}_curves.npz", idx=np.array(idx), grid=deletion_grid(N),
                        names=np.array(sorted(cur)), curves=np.stack(curves).astype(np.float32))
    Path(f"{stem}.json").write_text(json.dumps({
        "provenance": provenance(dataset=args.dataset, variant=variant, set=args.set,
                                 checkpoint=spec.checkpoint(variant).name,
                                 checkpoint_md5=md5(spec.checkpoint(variant)),
                                 test_accuracy_full=acc, n_images=len(idx), indices=idx, limit=args.limit,
                                 excluded={"pilot": len(pilot), "main": len(main_idx), "misclassified": len(mis_idx)},
                                 seconds=round(time.time() - t0, 1)),
        "calibration": cal, "records": records}))
    print(f"wrote {stem}.json")


if __name__ == "__main__":
    main()
