#!/usr/bin/env python3
"""
Randomization sanity check (descriptive; PROTOCOL.md).

Cascading re-initialization, top layer first: routing weights W_ij, then the primary
conv, then conv1. Each layer is replaced with Gaussian noise matching the original
tensor's standard deviation (seed: protocol.SEED). On the first
protocol.RANDOMIZATION_PER_CLASS main-set images of every class we report, per method, the Spearman correlation between the attribution map of the
trained model and that of the randomized model. A method that explains the model
should change when the model is destroyed.

Also recorded: Taylor's agreement with exact ablation on the randomized model, and
the class predicted by the trained and by each randomized model (attributions are
computed for each model's own prediction).
"""

import argparse
import copy
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from common import RESULTS, main_set, md5, provenance, set_determinism    # noqa: I100

from explorer import attribution as AT                                     # noqa: E402
from explorer import protocol as P                                         # noqa: E402
from explorer.registry import DATASETS                                     # noqa: E402

METHODS = P.SANITY_METHODS
STAGES = [("routing W", ["digits.routing_module.W"]),
          ("+ primary conv", ["primary.conv.weight", "primary.conv.bias"]),
          ("+ conv1", ["conv1.weight", "conv1.bias"])]


def randomize(model, names, gen):
    sd = model.state_dict()
    for n in names:
        t = sd[n]
        sd[n] = torch.randn(t.shape, generator=gen, device="cpu").to(t) * t.float().std()
    model.load_state_dict(sd)


def maps(model, x):
    r = AT.explain(model, x)
    return {"pred": r["pred"], **{m: r[m] for m in METHODS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--per_class", type=int, default=P.RANDOMIZATION_PER_CLASS)
    args = ap.parse_args()
    set_determinism()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = DATASETS[args.dataset]
    trained = spec.build_model(device)
    ds = spec.dataset(train=False)
    targets = spec.targets(ds)
    full = main_set(targets)
    n_cls = len(spec.classes)
    idx = [i for c in range(n_cls) for i in full[c * P.N_PER_CLASS: c * P.N_PER_CLASS + args.per_class]]

    gen = torch.Generator().manual_seed(P.SEED)
    models, m = [], copy.deepcopy(trained)
    for name, params in STAGES:
        randomize(m, params, gen)
        models.append((name, copy.deepcopy(m)))

    rows = []
    for i in idx:
        x = ds[i][0].unsqueeze(0).to(device)
        ref = maps(trained, x)
        rec = {"idx": int(i), "pred_trained": ref["pred"], "stages": {}}
        for name, rm in models:
            rnd = maps(rm, x)
            rec["stages"][name] = {
                "similarity": {k: float(spearmanr(ref[k], rnd[k])[0]) for k in METHODS},
                "taylor_vs_exact_randomized": float(spearmanr(rnd["taylor"], rnd["exact"])[0]),
                "pred": rnd["pred"],
            }
        rows.append(rec)

    summary = {}
    for name, _ in STAGES:
        summary[name] = {k: float(np.nanmedian([r["stages"][name]["similarity"][k] for r in rows]))
                         for k in METHODS}
        summary[name]["taylor_vs_exact_randomized"] = float(np.nanmedian(
            [r["stages"][name]["taylor_vs_exact_randomized"] for r in rows]))
    out = RESULTS / "sanity" / f"{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provenance": provenance(dataset=args.dataset, indices=idx, per_class=args.per_class,
                                                        seed=P.SEED, checkpoint_md5=md5(spec.checkpoint()),
                                                        stages=[n for n, _ in STAGES]),
                               "median_similarity": summary, "records": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
