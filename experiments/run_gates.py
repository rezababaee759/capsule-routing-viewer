#!/usr/bin/env python3
"""
Gate analysis on the reference run of each dataset, in double precision.

The network is rebuilt from the raw state_dict in float64 (convolutions, capsule reshaping,
squash, dynamic routing), independent of the app's code path; exact ablation is a plain loop
and the Taylor term comes from autograd on that graph. On GATE_N_PER_DATASET main-set images
per dataset it records:
  * the capsule gate f_i(α) = φ(u with u_i scaled by α), routing re-run and ĵ fixed, for the
    capsules at true ranks 1, k, k+1 and ⌈TOP_SET_FRAC·N⌉; Δ_i = f_i(1) − f_i(0) is the mean
    slope over [0, 1] and the Taylor estimate f_i'(1) the slope at α = 1;
  * the gap between the true k-th and (k+1)-th effects at the top-1% cutoff and the Taylor
    error around it, the pairs across the cutoff that Taylor orders correctly, and its misses;
  * P@1% of the Taylor estimate and of the exact effects under multiplicative noise
    (GATE_NOISE_SIGMAS), and the log-log fit of Taylor against Δ;
  * the recomputed predictions, P@1% and rank correlations next to the stored records.
Writes results/gates.json.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

from common import RESULTS, md5, provenance, set_determinism               # noqa: I100

# Shared DEFINITIONS only (protocol constants, training configuration); no app code path.
from explorer import protocol as P                                           # noqa: E402
from explorer.registry import DATASETS, ROUTING_ITERS                       # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D64 = torch.float64


# ----------------------------------------------------------------- independent network
class Net64:
    def __init__(self, state, n_types, iters=ROUTING_ITERS):
        g = lambda k: state[k].to(DEV, D64)
        self.c1w, self.c1b = g("conv1.weight"), g("conv1.bias")
        self.pw, self.pb = g("primary.conv.weight"), g("primary.conv.bias")
        self.W = g("digits.routing_module.W")[0]            # [K, N, 16, 8]
        self.T, self.iters = n_types, iters

    def primary(self, x):
        h = F.relu(F.conv2d(x.to(DEV, D64), self.c1w, self.c1b))
        p = F.conv2d(h, self.pw, self.pb, stride=2)[0]      # [T*8, G, G]
        G = p.shape[-1]
        p = p.view(self.T, 8, G, G)                          # channel = type*8 + dim
        u = torch.stack([p[t, :, gy, gx] for gy in range(G) for gx in range(G)
                         for t in range(self.T)])            # index = (gy*G + gx)*T + t
        return self.squash(u)

    @staticmethod
    def squash(s):
        n2 = (s * s).sum(-1, keepdim=True)
        return n2 / (1 + n2) * s / torch.sqrt(n2).clamp_min(1e-300)

    def route(self, u):
        """u: [N, 8] -> v: [K, 16]. b starts at 0; no update after the last iteration."""
        uhat = torch.einsum("knij,nj->kni", self.W, u)
        b = torch.zeros(uhat.shape[:2], dtype=D64, device=DEV)
        for t in range(self.iters):
            c = torch.softmax(b, dim=0)
            v = self.squash((c.unsqueeze(-1) * uhat).sum(1))
            if t < self.iters - 1:
                b = b + torch.einsum("kni,ki->kn", uhat, v)
        return v


def exact_loop(net, u, j):
    with torch.no_grad():
        phi0 = net.route(u)[j].norm()
        out = torch.empty(u.shape[0], dtype=D64)
        for i in range(u.shape[0]):
            ua = u.clone(); ua[i] = 0
            out[i] = phi0 - net.route(ua)[j].norm()
    return out.numpy()


def gate(net, u, i, j, alphas):
    """f_i(α) = ‖v_j‖ with u_i scaled by α, routing re-run, for every α."""
    out = np.empty(len(alphas))
    with torch.no_grad():
        for n, a in enumerate(alphas):
            ua = u.clone(); ua[i] = a * u[i]
            out[n] = float(net.route(ua)[j].norm())
    return out


def taylor64(net, u, j):
    u = u.clone().requires_grad_(True)
    phi = net.route(u)[j].norm()
    g, = torch.autograd.grad(phi, u)
    return (g * u).sum(-1).detach().cpu().numpy()


# ----------------------------------------------------------------- independent metrics
def p_at(scores, ref, frac):
    k = int(np.ceil(frac * len(ref)))
    a = np.argpartition(-scores, k - 1)[:k]           # different code path from argsort
    b = np.argpartition(-ref, k - 1)[:k]
    return np.intersect1d(a, b).size / k


def gate_measures(net, u, j, ex, ty, k):
    """Why the top-k set survives: the margin at the cutoff, Taylor's error around it, the
    misses, and the gates of the capsules at true ranks 1, k, k+1 and ⌈TOP_SET_FRAC·N⌉."""
    N = len(ex)
    o = np.argsort(-ex, kind="stable")
    rank = np.empty(N, int); rank[o] = np.arange(1, N + 1)
    win = o[int(np.ceil(k / 2)) - 1: k + int(np.ceil(k / 2))]          # true ranks ⌈k/2⌉ … k + ⌈k/2⌉
    assert (ex[win] > 0).all() and (ty[win] > 0).all()
    r = ty / np.where(ex != 0, ex, np.nan)
    inside, outside = win[rank[win] <= k], win[rank[win] > k]
    pairs = (ty[inside][:, None] > ty[outside][None, :]).mean()
    topt = set(np.argsort(-ty, kind="stable")[:k])
    misses = [{"true_rank": int(rank[i]), "ratio": float(r[i]),
               "gap_to_next": float(ex[i] / ex[o[rank[i]]])} for i in o[:k] if i not in topt]
    alphas = np.linspace(0, 1, P.GATE_STEPS)
    h = 1e-6
    gates = {}
    for name, rk in (("rank1", 1), ("rank_k", k), ("rank_k1", k + 1),
                     ("rank_topset", int(np.ceil(P.TOP_SET_FRAC * N)))):
        i = int(o[rk - 1])
        assert ex[i] > 0
        f = gate(net, u, i, j, alphas)
        fm, fp = gate(net, u, i, j, [1 - h, 1 + h])
        chord = f[0] + (f[-1] - f[0]) * alphas
        gates[name] = {"true_rank": rk, "exact": float(ex[i]), "taylor": float(ty[i]), "f": f.tolist(),
                       "chord_dev": float(np.abs(f - chord).max() / ex[i]),
                       "slope_at_1_fd": float((fp - fm) / (2 * h))}
    return {"cutoff_gap_ratio": float(ex[o[k - 1]] / ex[o[k]]),
            "cutoff_ratio_median": float(np.median(r[win])),
            "cutoff_logratio_sd": float(np.std(np.log(r[win]))),
            "boundary_pairs_ordered": float(pairs),
            "ratio_top1": float(r[o[0]]),
            "misses": misses, "gates": gates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=P.GATE_N_PER_DATASET, help="images per dataset")
    args = ap.parse_args()
    set_determinism()
    report = {}
    for key, spec in DATASETS.items():
        run = json.loads((RESULTS / "fidelity" / f"{key}_{P.REFERENCE_VARIANT[key]}_main.json").read_text())
        recs = run["records"]
        state = torch.load(spec.checkpoint(), map_location="cpu")
        net = Net64(state, spec.n_types)
        ds = spec.dataset()
        pick = np.random.default_rng(P.GATE_PICK_SEED).choice(len(recs), args.n, replace=False)
        rows = []
        for p in pick:
            rec = recs[p]
            x = ds[rec["idx"]][0].unsqueeze(0)
            u = net.primary(x)
            with torch.no_grad():
                v = net.route(u)
            j = int(v.norm(dim=-1).argmax())
            ex = exact_loop(net, u, j)
            ty = taylor64(net, u, j)
            N = len(ex)
            # ties: exactly equal values among the exact effects (float64)
            uniq = np.unique(ex).size
            srt = np.sort(ex)[::-1]
            k1 = int(np.ceil(P.TOPK_FRACS[0] * N))
            # noise injection: multiplicative log-normal error on the EXACT values
            rng = np.random.default_rng(int(rec["idx"]))
            noise = {}
            for sig in P.GATE_NOISE_SIGMAS:
                noisy = ex * np.exp(rng.normal(0, sig, N))
                noise[str(sig)] = p_at(noisy, ex, P.TOPK_FRACS[0])
            # Is Taylor's error systematic (monotone in Δ) or random? Log-log fit on the
            # true top 5% (positive effects): R² near 1 means a smooth distortion, which
            # preserves the ranking even when magnitudes are off.
            top5 = np.argsort(-ex)[:int(np.ceil(P.TOP_SET_FRAC * N))]
            pos = top5[(ex[top5] > 0) & (ty[top5] > 0)]
            lx, ly = np.log(ex[pos]), np.log(ty[pos])
            slope, icpt = np.polyfit(lx, ly, 1)
            r2 = 1 - np.sum((ly - (slope * lx + icpt)) ** 2) / np.sum((ly - ly.mean()) ** 2)
            # random-noise level that would match Taylor's observed log-residual spread
            resid_sd = float(np.std(ly - (slope * lx + icpt)))
            row = {
                "loglog_slope_top5": float(slope), "loglog_r2_top5": float(r2),
                "loglog_resid_sd_top5": resid_sd,
                "idx": rec["idx"], "pred_stored": rec["pred"], "pred_independent": j,
                "phi_stored": rec["phi"], "phi_independent": float(v[j].norm()),
                "p1_independent": p_at(ty, ex, P.TOPK_FRACS[0]), "p1_stored": rec["rank"]["taylor"]["p1"],
                "tau_independent": float(kendalltau(ty, ex)[0]), "tau_stored": rec["rank"]["taylor"]["tau"],
                "rho_independent": float(spearmanr(ty, ex)[0]), "rho_stored": rec["rank"]["taylor"]["rho"],
                "unique_exact_values": int(uniq), "N": N,
                "frac_exact_abs_below_1e-9": float(np.mean(np.abs(ex) < 1e-9)),
                "rel_err_taylor_top1pct": float(np.median(np.abs(ty[np.argsort(-ex)[:k1]] - srt[:k1]) / srt[:k1])),
                "noise_p1": noise,
                **gate_measures(net, u, j, ex, ty, k1),
            }
            rows.append(row)
            print(f"{key:13s} idx {rec['idx']:5d} | pred {j}=={rec['pred']} φ {row['phi_independent']:.6f} vs "
                  f"{rec['phi']:.6f} | P@1% indep {row['p1_independent']:.3f} stored {row['p1_stored']:.3f} | "
                  f"τ {row['tau_independent']:.4f}/{row['tau_stored']:.4f} | top-1%-rel-err "
                  f"{row['rel_err_taylor_top1pct']:.2f} | log-log R² {r2:.4f} slope {slope:.3f} resid sd {resid_sd:.3f} | "
                  f"noise P@1% {noise}", flush=True)
        report[key] = {
            "rows": rows,
            "max_abs_diff": {k: float(max(abs(r[f"{k}_independent"] - r[f"{k}_stored"]) for r in rows))
                             for k in ("p1", "tau", "rho", "phi")},
            "pred_agreement": float(np.mean([r["pred_independent"] == r["pred_stored"] for r in rows])),
        }
        print(f"  => {key}: max |independent − stored|: {report[key]['max_abs_diff']}, "
              f"pred agreement {report[key]['pred_agreement']}")
    out = RESULTS / "gates.json"
    report["provenance"] = provenance(n=args.n, pick_seed=P.GATE_PICK_SEED, sigmas=list(P.GATE_NOISE_SIGMAS),
                                      gate_steps=P.GATE_STEPS,
                                      inputs={k: md5(RESULTS / "fidelity" / f"{k}_{P.REFERENCE_VARIANT[k]}_main.json") for k in DATASETS})
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
