#!/usr/bin/env python3
"""
Choice of score: capsule length against the decision margin.

The fidelity runs score a capsule by its effect on φ = ‖v_ĵ‖. Here the same main-set
images are also scored by the margin m = ‖v_ĵ‖ − max_{j≠ĵ} ‖v_j‖ (ĵ the intact
prediction; the runner-up is re-chosen after every ablation). Per image:
  * exact margin effects Δ^m_i = m(u) − m(u, u_i = 0) and the Taylor margin estimate
    ⟨∂m/∂u_i, u_i⟩ (gradient with the intact runner-up r, i.e. of ‖v_ĵ‖ − ‖v_r‖);
  * how Taylor-margin ranks against exact margin, how the exact length and exact margin
    rankings agree, and how Taylor-length ranks against exact margin;
  * the share of single ablations that change the runner-up or the prediction;
  * the (exact, Taylor) length pairs on the true top TOP_SET_FRAC, for the per-capsule
    error distribution.

    python experiments/run_margin.py --dataset mnist --variant seed42

Writes results/margin/<dataset>_<variant>_main.json (one record per image).
"""

import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import kendalltau

from common import RESULTS, main_set, md5, provenance, set_determinism       # noqa: I100
from run_fidelity import rank_metrics, topk                                  # noqa: E402

from explorer import attribution as AT                                       # noqa: E402
from explorer import protocol as P                                           # noqa: E402
from explorer.registry import DATASETS                                       # noqa: E402


def readouts(v, j):
    """Per row: φ, the margin, the runner-up and the predicted class."""
    lengths = v.norm(dim=-1)
    others = lengths.clone()
    others[:, j] = -torch.inf
    return torch.stack([AT.confidence(v, j), AT.margin(v, j), others.argmax(dim=1).to(v.dtype),
                        lengths.argmax(dim=1).to(v.dtype)], dim=1)


@torch.no_grad()
def single_ablations(model, u, j, chunk=128):
    """Readouts of the intact input and of every single-capsule ablation: ([4], [N, 4])."""
    N = u.shape[1]
    keep = torch.ones(N, N, device=u.device)
    keep[torch.arange(N), torch.arange(N)] = 0.0
    return readouts(model.digits(u)[0], j)[0], AT.phi_of_masks(model, u, keep, j, chunk=chunk, readout=readouts)


def taylor_margin(model, u, j, r):
    """⟨∂(‖v_j‖ − ‖v_r‖)/∂u_i, u_i⟩ for every capsule, through the unrolled routing."""
    u = u.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        v, *_ = model.digits(u)
        lengths = v[0].norm(dim=-1)
        g, = torch.autograd.grad(lengths[j] - lengths[r], u)
    return (g[0] * u[0].detach()).sum(-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--variant", default=None, choices=list(P.VARIANTS), help="default: the reference run")
    ap.add_argument("--limit", type=int, default=None, help="debug: first n images only")
    args = ap.parse_args()
    args.variant = args.variant or P.REFERENCE_VARIANT[args.dataset]
    set_determinism()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASETS[args.dataset]
    model = spec.build_model(device, args.variant)
    ds = spec.dataset(train=False)
    targets = spec.targets(ds)
    idx = main_set(targets)
    if args.limit:
        idx = idx[:args.limit]
    stored = RESULTS / "fidelity" / f"{args.dataset}_{args.variant}_main.json"
    stored_p1 = {r["idx"]: r["rank"]["taylor"]["p1"] for r in json.loads(stored.read_text())["records"]}

    records, t0 = [], time.time()
    for n, i in enumerate(idx):
        x = ds[i][0].unsqueeze(0).to(device)
        u = AT.primary_capsules(model, x)
        j = AT.predicted_class(model, u)
        intact, abl = single_ablations(model, u, j)
        ex_len = (intact[0] - abl[:, 0]).cpu().numpy()
        ex_mar = (intact[1] - abl[:, 1]).cpu().numpy()
        r0 = int(intact[2])
        ta_len = AT.gradient_attributions(model, u, j)[0]["taylor"]
        ta_mar = taylor_margin(model, u, j, r0)
        top5 = topk(ex_len, P.TOP_SET_FRAC)
        k1 = topk(ex_len, P.TOPK_FRACS[0])
        rec = {"idx": int(i), "label": int(targets[i]), "pred": j, "correct": bool(j == targets[i]),
               "margin": float(intact[1]), "runner_up": r0,
               "taylor_margin_vs_exact_margin": rank_metrics(ta_mar, ex_mar),
               "exact_length_vs_exact_margin": {
                   "overlap_top1": len(set(k1) & set(topk(ex_mar, P.TOPK_FRACS[0]))) / len(k1),
                   "tau": float(kendalltau(ex_len, ex_mar)[0])},
               "taylor_length_vs_exact_margin": rank_metrics(ta_len, ex_mar),
               "taylor_length_p1": rank_metrics(ta_len, ex_len)["p1"],
               "share_runner_up_changes": float((abl[:, 2] != intact[2]).float().mean()),
               "share_prediction_changes": float((abl[:, 3] != j).float().mean()),
               "top_set_pairs": {"exact": ex_len[top5].tolist(), "taylor": ta_len[top5].tolist()}}
        rec["matches_stored_taylor_p1"] = bool(rec["taylor_length_p1"] == stored_p1[int(i)])
        records.append(rec)
        if (n + 1) % 50 == 0 or n + 1 == len(idx):
            p1 = np.mean([q["taylor_margin_vs_exact_margin"]["p1"] for q in records])
            print(f"  {n + 1}/{len(idx)}  {time.time() - t0:.0f}s  (mean Taylor-margin P@1% so far {p1:.3f})",
                  flush=True)

    out = RESULTS / "margin" / f"{args.dataset}_{args.variant}_main.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "provenance": provenance(dataset=args.dataset, variant=args.variant, set="main",
                                 checkpoint=spec.checkpoint(args.variant).name,
                                 checkpoint_md5=md5(spec.checkpoint(args.variant)), n_images=len(idx),
                                 indices=idx, limit=args.limit, main_set_seed=P.SEED,
                                 inputs={str(stored.relative_to(RESULTS.parent)): md5(stored)},
                                 seconds=round(time.time() - t0, 1)),
        "records": records}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
