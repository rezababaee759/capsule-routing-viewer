#!/usr/bin/env python3
"""
EQ2: cost per image (descriptive; PROTOCOL.md). Batch size 1, warm-up, then timed
repetitions; median and IQR in milliseconds, on GPU and on CPU, for each dataset's
reference run. Also the app-side costs: Python time to analyse one sample and
build the 3D page, and the size of that page.
"""

import json
import time

import numpy as np
import torch

from common import RESULTS, main_set, md5, provenance, set_determinism            # noqa: I100

from explorer import attribution as AT                                       # noqa: E402
from explorer import data as D                                               # noqa: E402
from explorer import hero as H                                               # noqa: E402
from explorer import protocol as P                                           # noqa: E402
from explorer.registry import DATASETS, MODELS                               # noqa: E402


def timeit(fn, device, warmup, reps):
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    q = np.percentile(ts, [25, 50, 75])
    return {"median_ms": float(q[1]), "iqr_ms": [float(q[0]), float(q[2])], "reps": reps}


def bench(spec, device, warmup, reps, reps_exact):
    model = spec.build_model(device)
    ds = spec.dataset()
    x = ds[main_set(spec.targets(ds))[0]][0].unsqueeze(0).to(device)
    u = AT.primary_capsules(model, x)
    j = AT.predicted_class(model, u)
    grads, _, _ = AT.gradient_attributions(model, u, j)

    def forward():
        with torch.no_grad():
            model(x)

    out = {
        "N": int(u.shape[1]),
        "forward": timeit(forward, device, warmup, reps),
        "gradient_attributions": timeit(lambda: AT.gradient_attributions(model, AT.primary_capsules(model, x)),
                                        device, warmup, reps),
        "exact_ablation": timeit(lambda: AT.exact_ablation(model, AT.primary_capsules(model, x), j),
                                 device, min(warmup, P.COST_WARMUP_CPU), reps_exact),
    }
    return out, model, x


def main():
    set_determinism()
    res = {}
    devices = [torch.device("cuda")] if torch.cuda.is_available() else []
    devices.append(torch.device("cpu"))
    for key, spec in DATASETS.items():
        res[key] = {}
        for dev in devices:
            gpu = dev.type == "cuda"
            r, model, x = bench(spec, dev,
                                     warmup=P.COST_WARMUP_GPU if gpu else P.COST_WARMUP_CPU,
                                     reps=P.COST_REPS_GPU if gpu else P.COST_REPS_CPU,
                                     reps_exact=P.COST_REPS_EXACT_GPU if gpu else P.COST_REPS_EXACT_CPU)
            if gpu:   # app-side: one sample's full analysis + building the 3D page
                t0 = time.perf_counter()
                tr = D.trace_routing(model, x, spec)
                attr = D.attributions(model, x)
                t1 = time.perf_counter()
                html = H.build_html(tr, attr, spec.classes)
                t2 = time.perf_counter()
                r["app"] = {"analysis_ms": (t1 - t0) * 1e3, "page_build_ms": (t2 - t1) * 1e3,
                            "page_mb": len(html.encode()) / 1e6}
            res[key][dev.type] = r
            print(key, dev.type, {k: v["median_ms"] for k, v in r.items() if isinstance(v, dict) and "median_ms" in v})
    out = RESULTS / "cost.json"
    out.write_text(json.dumps({"provenance": provenance(
        cpu_threads=torch.get_num_threads(), cpu=__import__("platform").processor() or "unknown",
        reps={"gpu": [P.COST_WARMUP_GPU, P.COST_REPS_GPU, P.COST_REPS_EXACT_GPU],
              "cpu": [P.COST_WARMUP_CPU, P.COST_REPS_CPU, P.COST_REPS_EXACT_CPU]},
        seed=P.SEED, checkpoints={k: md5(DATASETS[k].checkpoint()) for k in P.DATASETS},
        note="one image per dataset (main_set[0]); app timings are a single run"),
                               "results": res}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
