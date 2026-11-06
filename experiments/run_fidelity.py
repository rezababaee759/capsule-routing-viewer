#!/usr/bin/env python3
"""
EQ1 / EQ3 / EQ4: per-image fidelity of every attribution method against exact
ablation, group-deletion behaviour, and routing-behaviour measures.

    python experiments/run_fidelity.py --dataset mnist --variant seed42 --set main
    python experiments/run_fidelity.py --dataset cifar10 --set misclassified   # reference run

Writes results/fidelity/<dataset>_<variant>_<set>.json (one record per image).
"""

import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from common import (RESULTS, main_set, md5, misclassified_set, pilot_indices, predictions,  # noqa: I100
                    provenance, set_determinism)

from explorer import analysis as A                                        # noqa: E402
from explorer import attribution as AT                                    # noqa: E402
from explorer import protocol as P                                        # noqa: E402
from explorer.registry import DATASETS, MODELS                            # noqa: E402


def topk(scores, frac):
    k = max(1, int(np.ceil(frac * len(scores))))
    return np.argsort(-scores, kind="stable")[:k]


def rank_metrics(s, ex):
    fa, fb = P.TOPK_FRACS
    topa, topb = set(topk(ex, fa)), set(topk(ex, fb))
    union = np.array(sorted(set(topk(s, P.TOP_SET_FRAC)) | set(topk(ex, P.TOP_SET_FRAC))))
    return {
        "rho": float(spearmanr(s, ex)[0]),
        "tau": float(kendalltau(s, ex)[0]),
        "p1": len(set(topk(s, fa)) & topa) / len(topa),
        "p5": len(set(topk(s, fb)) & topb) / len(topb),
        "rho_top": float(spearmanr(s[union], ex[union])[0]),
    }


def magnitude_metrics(est, ex):
    idx = topk(ex, P.TOP_SET_FRAC)
    e, t = est[idx], ex[idx]
    return {"rel_l1_top5": float(np.abs(e - t).sum() / np.abs(t).sum()),
            "slope_top5": float((e @ t) / (t @ t))}


def deletion_curves(model, u, j, rankings, N):
    """Deletion-only curves for many rankings in one batched pass. Returns ks and {m: φ/φ0}."""
    ks = AT.budgets(N, P.CURVE_STEPS)
    budget_ks = [max(1, int(round(b * N))) for b in P.BUDGETS]
    ks = np.unique(np.concatenate([ks, budget_ks]))
    masks, owners = [], []
    for m, scores in rankings.items():
        rank = np.empty(N, int)
        rank[np.argsort(-scores, kind="stable")] = np.arange(N)
        masks.append(rank[None, :] >= ks[:, None])                    # keep = not among top-k
        owners.append(m)
    keep = torch.as_tensor(np.concatenate(masks), device=u.device)
    phi = AT.phi_of_masks(model, u, keep, j, chunk=96).cpu().numpy().reshape(len(owners), len(ks))
    return ks, {m: phi[i] / max(phi[i][0], 1e-12) for i, m in enumerate(owners)}


def curve_summary(ks, y, N):
    # The endpoint k = N (every capsule removed) is φ = 0 by construction, so the
    # minimum and rebound are taken over partial deletions k < N.
    yp = y[ks < N]
    i_min = int(np.argmin(yp))
    out = {"min": float(yp[i_min]), "argmin_frac": float(ks[i_min] / N),
           "rebound": float(yp[i_min:].max() - yp[i_min]),
           "auc_log": float(np.trapz(y, np.log1p(ks)) / np.log1p(N))}
    for b in P.BUDGETS:
        out[f"del_{int(b * 100)}"] = float(y[np.searchsorted(ks, max(1, int(round(b * N))))])
    return out


def routing_behaviour(model, u, j, top):
    """Per routing iteration: mean normalized entropy (all capsules and the exact top 1%),
    consensus = fraction of capsules whose coupling to ĵ is STRICTLY the largest
    (ties at uniform count as no preference), and ‖v_ĵ‖."""
    with torch.no_grad():
        _, _, _, hist = model.digits(u)
    ent, ent_top, cons, vlen = [], [], [], []
    for h in hist:
        c = h["coupling_coeffs"][0].cpu().numpy()
        H = A.normalized_entropy(c)
        others = np.delete(c, j, axis=0).max(axis=0)
        ent.append(float(H.mean()))
        ent_top.append(float(H[top].mean()))
        cons.append(float((c[j] > others + P.TIE_EPS).mean()))
        vlen.append(float(np.linalg.norm(h["digit_capsules"][0, j].cpu().numpy())))
    return {"entropy_iter": ent, "entropy_top1_iter": ent_top, "consensus_iter": cons,
            "vlen_iter": vlen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--variant", default=None, choices=list(P.VARIANTS), help="default: the reference run")
    ap.add_argument("--set", default="main", choices=["main", "misclassified"])
    ap.add_argument("--limit", type=int, default=None, help="debug: first n images only")
    args = ap.parse_args()
    args.variant = args.variant or P.REFERENCE_VARIANT[args.dataset]
    set_determinism()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASETS[args.dataset]
    model = spec.build_model(device, args.variant)
    ds = spec.dataset(train=False)
    targets = spec.targets(ds)

    main_idx = main_set(targets)
    if args.set == "main":
        idx = main_idx
        acc = float((predictions(model, ds, device) == np.asarray(targets)).mean())
    else:
        idx, acc = misclassified_set(model, ds, targets, device,
                                     exclude=set(main_idx) | pilot_indices(len(targets)))
    if args.limit:
        idx = idx[:args.limit]

    records, t0 = [], time.time()
    for n, i in enumerate(idx):
        x = ds[i][0].unsqueeze(0).to(device)
        r = AT.explain(model, x, seed=int(i))
        u = AT.primary_capsules(model, x)
        N, j, ex = u.shape[1], r["pred"], r["exact"]
        methods = [m for m in P.METHODS if m in r]
        rec = {"idx": int(i), "label": int(targets[i]), "pred": j, "correct": bool(j == targets[i]),
               "phi": r["phi"], "routing_share": r["routing_share"],
               "concentration_top1": float(ex[topk(ex, P.TOPK_FRACS[0])].sum() / max(ex[ex > 0].sum(), 1e-12)),
               **routing_behaviour(model, u, j, topk(ex, P.TOPK_FRACS[0])),
               "rank": {m: rank_metrics(r[m], ex) for m in methods},
               "magnitude": {m: magnitude_metrics(r[m], ex) for m in P.MAGNITUDE_METHODS if m in r}}
        ks, curves = deletion_curves(model, u, j, {"exact": ex, **{m: r[m] for m in methods}}, N)
        rec["deletion"] = {m: curve_summary(ks, y, N) for m, y in curves.items()}
        if n < P.STORED_CURVES:                             # keep a few full curves for inspection
            rec["curves"] = {"ks": ks.tolist(), **{m: np.round(y, 5).tolist() for m, y in curves.items()}}
        records.append(rec)
        if (n + 1) % 25 == 0 or n + 1 == len(idx):
            el = time.time() - t0
            p1 = np.median([q["rank"]["taylor"]["p1"] for q in records])
            print(f"  {n + 1}/{len(idx)}  {el:.0f}s  (median taylor P@1% so far {p1:.3f})", flush=True)

    out = RESULTS / "fidelity" / f"{args.dataset}_{args.variant}_{args.set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "provenance": provenance(dataset=args.dataset, variant=args.variant, set=args.set,
                                 checkpoint=spec.checkpoint(args.variant).name,
                                 checkpoint_md5=md5(spec.checkpoint(args.variant)),
                                 test_accuracy_full=acc, n_images=len(idx), indices=idx, limit=args.limit,
                                 main_set_seed=P.SEED, pilot_seed=P.PILOT_SEED,
                                 misclassified_seed=P.SEED + P.MISCLASSIFIED_SEED_OFFSET,
                                 seconds=round(time.time() - t0, 1)),
        "records": records}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
