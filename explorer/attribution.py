"""
Routing-aware capsule attribution.

Target quantity. For an input x with predicted class ĵ, the confidence is the
length of the winning digit capsule,  φ(u) = ‖v_ĵ(u)‖,  where u ∈ R^{N×8} are the
primary capsules and v = routing(u) runs all routing iterations. The importance
of capsule i is its exact ablation effect

    Δ_i = φ(u) − φ(u with u_i = 0),

which requires N extra routing passes. Gradient attributions estimate Δ from a
single backward pass through the *unrolled* routing:

    taylor_i  = ⟨∂φ/∂u_i, u_i⟩            first-order Taylor estimate of Δ_i
    g_A_i     = ‖∂φ/∂u_i‖                 sensitivity to the capsule's pose
    g_C_i     = |∂φ/∂c_iĵ|                sensitivity to its coupling to ĵ

`taylor_frozen` repeats the Taylor estimate with the final couplings held
constant, i.e. ignoring how routing would react; comparing the two isolates the
contribution of routing to the attribution.
"""

import numpy as np
import torch

from capsule_core import squash

from .protocol import CURVE_STEPS



# ---------------------------------------------------------------- forward helpers
def primary_capsules(model, x):
    with torch.no_grad():
        return model.primary(torch.relu(model.conv1(x)))          # [B, N, 8]


def confidence(v, j):
    """φ = ‖v_j‖ for each batch row."""
    return v[:, j].norm(dim=-1)


def margin(v, j):
    """‖v_j‖ minus the longest other class capsule, for each batch row."""
    lengths = v.norm(dim=-1)
    others = lengths.clone()
    others[:, j] = -torch.inf
    return lengths[:, j] - others.max(dim=1).values


def predicted_class(model, u):
    with torch.no_grad():
        v, *_ = model.digits(u)
    return int(v[0].norm(dim=-1).argmax())


# ---------------------------------------------------------------- gradients
def gradient_attributions(model, u, j=None):
    """All gradient attributions for one image. u: [1, N, 8]. Returns numpy [N] arrays."""
    if j is None:
        j = predicted_class(model, u)
    u = u.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        v, c, _, _ = model.digits(u)                               # c: final couplings [1, K, N]
        phi = confidence(v, j)[0]
        g_u, g_c = torch.autograd.grad(phi, [u, c])

        # frozen routing: same final couplings, but treated as constants
        u2 = u.detach().clone().requires_grad_(True)
        W = model.digits.routing_module.W
        u_hat = torch.matmul(W, u2.unsqueeze(1).unsqueeze(-1)).squeeze(-1)   # [1, K, N, 16]
        s = (c.detach().unsqueeze(-1) * u_hat).sum(dim=2)
        phi_frozen = squash(s)[0, j].norm()
        g_u_frozen, = torch.autograd.grad(phi_frozen, [u2])

    u0, gu, gc, cc = u[0].detach(), g_u[0], g_c[0], c[0].detach()       # [N,8] [N,8] [K,N] [K,N]
    # Share of the input gradient that comes from routing (the c-dependence on u):
    # ‖∂φ/∂u − ∂φ/∂u|_c‖ / ‖∂φ/∂u‖ over all capsules.
    routing_share = float((gu - g_u_frozen[0]).norm() / gu.norm().clamp_min(1e-12))
    out = {
        "taylor": (gu * u0).sum(-1),
        "taylor_frozen": (g_u_frozen[0] * u0).sum(-1),
        "g_A": gu.norm(dim=-1),
        "g_C": gc[j].abs(),
    }
    res = {k: t.detach().cpu().numpy() for k, t in out.items()}
    res["routing_share"] = routing_share          # scalar, not a per-capsule map
    return res, j, float(phi)


# ---------------------------------------------------------------- exact ablation
@torch.no_grad()
def phi_of_masks(model, u, keep, j, chunk=128, readout=confidence):
    """readout(v, j) (default φ) for a batch of keep-masks. u: [1, N, 8]; keep: [M, N]
    bool/float. Returns [M], or [M, ...] for a readout with more outputs per row."""
    out = []
    for s in range(0, keep.shape[0], chunk):
        m = keep[s:s + chunk].to(u.dtype).unsqueeze(-1)            # [B, N, 1]
        v, *_ = model.digits(u.expand(m.shape[0], -1, -1) * m)
        out.append(readout(v, j))
    return torch.cat(out)


@torch.no_grad()
def exact_ablation(model, u, j, chunk=128, readout=confidence):
    """Δ_i = s(u) − s(u, u_i = 0) for every capsule, with s = readout (default φ). Returns numpy [N]."""
    N = u.shape[1]
    phi0 = readout(model.digits(u)[0], j)[0]
    out = torch.empty(N, device=u.device, dtype=u.dtype)
    for s in range(0, N, chunk):
        idx = torch.arange(s, min(N, s + chunk), device=u.device)
        keep = torch.ones(len(idx), N, device=u.device)
        keep[torch.arange(len(idx)), idx] = 0.0
        out[idx] = phi0 - phi_of_masks(model, u, keep, j, chunk=chunk, readout=readout)
    return out.cpu().numpy()


# ---------------------------------------------------------------- frozen couplings
@torch.no_grad()
def frozen_votes(model, u, j):
    """Contributions w_i = c*_iĵ û_ĵ|i to s_ĵ, with c* the final couplings of the intact
    input. With c* held fixed, s_ĵ(S removed) = Σ_{i∉S} w_i. Returns [N, 16]."""
    _, c, _, _ = model.digits(u)
    W = model.digits.routing_module.W[0, j]                         # [N, 16, 8]
    u_hat = torch.matmul(W, u[0].unsqueeze(-1)).squeeze(-1)         # [N, 16]
    return c[0, j].unsqueeze(-1) * u_hat


def frozen_deletion(w, order):
    """‖s_ĵ‖, φ = ‖squash(s_ĵ)‖ and the cosine between s_ĵ and its intact direction, with
    couplings held fixed, after removing the first k capsules of `order`, for k = 0..N.
    w: [N, 16] from frozen_votes. Returns float64 numpy. Runs on the CPU, where the
    cumulative sum is deterministic."""
    w = w.double().cpu()
    s = w.sum(0) - torch.cat([w.new_zeros(1, w.shape[1]), w[torch.as_tensor(order)].cumsum(0)])
    norm = s.norm(dim=-1)
    cos = torch.where(norm > 0, (s @ s[0]) / (norm * norm[0]).clamp_min(1e-300), torch.nan)
    return norm.numpy(), squash(s).norm(dim=-1).numpy(), cos.numpy()


def frozen_ablation(w):
    """Single-capsule effects with couplings held fixed: Δ^F_i = φ_F(u) − φ_F(u, u_i = 0)."""
    w = w.double().cpu()
    s0 = w.sum(0)
    return (squash(s0).norm() - squash(s0 - w).norm(dim=-1)).numpy()


# ---------------------------------------------------------------- baselines
@torch.no_grad()
def baseline_attributions(model, u, j, seed=0):
    """Non-gradient baselines. `seed` only drives the random ranking (the app uses
    the default; the experiments pass the image index so each image gets its own)."""
    v, c, _, _ = model.digits(u)
    a = u[0].norm(dim=-1)
    cj = c[0, j]
    rng = np.random.default_rng(seed)
    return {
        "activation": a.cpu().numpy(),
        "coupling": cj.cpu().numpy(),
        "act_x_coupling": (a * cj).cpu().numpy(),
        "random": rng.random(u.shape[1]),
    }


# ---------------------------------------------------------------- faithfulness curves
def budgets(N, steps=CURVE_STEPS):
    """Capsule counts 0..N, log-spaced so small budgets (where deletion is
    informative) are resolved finely."""
    return np.unique(np.concatenate([[0], np.geomspace(1, N, steps).round()]).astype(int))


def deletion_insertion(model, u, j, scores, steps=CURVE_STEPS, chunk=64, ks=None):
    """
    Remove (deletion) or add (insertion) capsules in decreasing-score order and
    re-route after each step. Returns fractions k/N, deletion φ/φ0, insertion φ/φ0.
    """
    N = u.shape[1]
    order = torch.as_tensor(np.argsort(-np.asarray(scores), kind="stable").copy(), device=u.device)
    ks = budgets(N, steps) if ks is None else np.asarray(ks, dtype=int)
    rank = torch.empty(N, device=u.device, dtype=torch.long)
    rank[order] = torch.arange(N, device=u.device)
    kt = torch.as_tensor(ks, device=u.device).unsqueeze(1)
    top = rank.unsqueeze(0) < kt                                    # [S, N] True = among top-k
    phi_del = phi_of_masks(model, u, ~top, j, chunk)
    phi_ins = phi_of_masks(model, u, top, j, chunk)
    phi0 = phi_del[0].clamp_min(1e-12)
    return ks / N, (phi_del / phi0).cpu().numpy(), (phi_ins / phi0).cpu().numpy()


def auc(x, y):
    return float(np.trapz(y, x))


# ---------------------------------------------------------------- everything for one image
def explain(model, x, with_curves=False, seed=0):
    """Full attribution bundle for one model input x: [1, C, H, W]."""
    u = primary_capsules(model, x)
    grads, j, phi = gradient_attributions(model, u)
    res = {"pred": j, "phi": phi, "exact": exact_ablation(model, u, j), **grads,
           **baseline_attributions(model, u, j, seed=seed)}
    if with_curves:
        res["curves"] = {k: deletion_insertion(model, u, j, res[k])
                         for k in ("exact", "taylor", "taylor_frozen", "g_A", "activation")}
    return res


GRADIENT_METHODS = ("taylor", "taylor_frozen", "g_A", "g_C")
BASELINE_METHODS = ("activation", "coupling", "act_x_coupling", "random")
