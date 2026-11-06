#!/usr/bin/env python3
"""
Per-capsule data for the figures (cached in results/cache/figdata_<dataset>.npz).

Images are taken from the protocol's main set by rule, never by eye:
  * deletion curves: the first FIG_CURVES_PER_CLASS main-set images of every class;
  * magnitude scatter: the first FIG_MAGNITUDE_PER_CLASS of every class, true top-TOP_SET_FRAC capsules;
  * case studies (reference run, main + misclassified sets):
      - "confident correct": correct prediction with the largest ‖v_ĵ‖,
      - "confident error":   misclassified-set image with the largest ‖v_ĵ‖,
      - "ambiguous":         correct prediction with the smallest top-2 length margin.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from common import main_set, md5, provenance, set_determinism             # noqa: E402
from explorer import protocol as P                                          # noqa: E402
from explorer import attribution as AT                                      # noqa: E402
from explorer.registry import DATASETS                                      # noqa: E402

CURVE_METHODS = list(P.DELETION_FIGURE_METHODS)


def main():
    set_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, spec in DATASETS.items():
        out = ROOT / "results" / "cache" / f"figdata_{key}.npz"
        model = spec.build_model(device)
        ds = spec.dataset()
        targets = spec.targets(ds)
        main_idx = np.array(main_set(targets)).reshape(len(spec.classes), P.N_PER_CLASS)
        N = None

        # deletion curves
        curves, ks = {m: [] for m in CURVE_METHODS}, None
        for i in main_idx[:, :P.FIG_CURVES_PER_CLASS].ravel():
            x = ds[int(i)][0].unsqueeze(0).to(device)
            r = AT.explain(model, x)
            u = AT.primary_capsules(model, x)
            N = u.shape[1]
            for m in CURVE_METHODS:
                f, d, _ = AT.deletion_insertion(model, u, r["pred"], r[m], steps=P.CURVE_STEPS)
                curves[m].append(d)
            ks = f
        # magnitude pairs
        mag = {"exact": [], "taylor": [], "taylor_frozen": []}
        for i in main_idx[:, :P.FIG_MAGNITUDE_PER_CLASS].ravel():
            x = ds[int(i)][0].unsqueeze(0).to(device)
            r = AT.explain(model, x)
            top = np.argsort(-r["exact"])[:int(np.ceil(P.TOP_SET_FRAC * len(r["exact"])))]
            for m in mag:
                mag[m].append(r[m][top])
        # case studies from the stored fidelity records (rule-based)
        recs = json.loads((ROOT / "results" / "fidelity" / f"{key}_{P.REFERENCE_VARIANT[key]}_main.json").read_text())["records"]
        mis = json.loads((ROOT / "results" / "fidelity" / f"{key}_{P.REFERENCE_VARIANT[key]}_misclassified.json").read_text())["records"]
        conf_correct = max((r for r in recs if r["correct"]), key=lambda r: r["phi"])
        conf_error = max(mis, key=lambda r: r["phi"])
        margins = []
        for r in recs:
            if not r["correct"]:
                continue
            with torch.no_grad():
                v, *_ = model(ds[r["idx"]][0].unsqueeze(0).to(device))
            L = np.sort(v[0].norm(dim=-1).cpu().numpy())[::-1]
            margins.append((L[0] - L[1], r["idx"]))
        ambiguous = min(margins)[1]
        cases = {"confident_correct": conf_correct["idx"], "confident_error": conf_error["idx"],
                 "ambiguous": ambiguous}
        case_metrics = {}
        for name, i in cases.items():
            x, label = ds[int(i)]
            r = AT.explain(model, x.unsqueeze(0).to(device))
            k = int(np.ceil(P.TOPK_FRACS[0] * len(r["exact"])))
            top_ex, top_ty = np.argsort(-r["exact"])[:k], np.argsort(-r["taylor"])[:k]
            i1 = int(top_ex[0])
            case_metrics[name] = {"idx": int(i), "label": int(label), "pred": int(r["pred"]), "phi": r["phi"],
                                  "p1": len(set(top_ex) & set(top_ty)) / k,
                                  "top1_exact": float(r["exact"][i1]), "top1_taylor": float(r["taylor"][i1]),
                                  "top1ratio": float(r["taylor"][i1] / r["exact"][i1])}
        np.savez_compressed(out, ks=ks, N=N,
                            **{f"curve_{m}": np.array(v) for m, v in curves.items()},
                            **{f"mag_{m}": np.concatenate(v) for m, v in mag.items()},
                            cases=json.dumps(cases), case_metrics=json.dumps(case_metrics),
                            provenance=json.dumps(provenance(
                                dataset=key, checkpoint_md5=md5(spec.checkpoint()),
                                inputs={s: md5(ROOT / "results" / "fidelity" / f"{key}_{P.REFERENCE_VARIANT[key]}_{s}.json")
                                        for s in ("main", "misclassified")})))
        print(f"{key}: curves {len(curves['exact'])} imgs, magnitude pairs {len(np.concatenate(mag['exact']))}, "
              f"cases {cases}", flush=True)


if __name__ == "__main__":
    main()
