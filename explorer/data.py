"""
A full trace of one forward pass through the CapsNet, plus its attributions.

Everything is recomputed from public attributes of ``CapsuleNetwork`` (conv1,
primary, digits.routing_module.W, decoder), so ``capsule_core.py`` stays untouched.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from capsule_core import squash

from . import attribution as AT
from .registry import geometry_of


def get_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_index(targets, cls=None, rng=None):
    """Random test index, optionally restricted to a given true class."""
    rng = rng or random.Random()
    if cls is None:
        return rng.randrange(len(targets))
    return rng.choice(np.flatnonzero(np.asarray(targets) == cls).tolist())


@torch.no_grad()
def trace_routing(model, x, spec):
    """
    Run one model input x [1, C, H, W] through the network and keep every
    intermediate quantity. Returns numpy arrays:
        image          [H, W] or [H, W, 3]   display image in [0, 1]
        conv1          [256, H1, W1]         ReLU feature maps
        u              [N, 8]                primary capsule vectors (squashed)
        u_norm         [N]
        u_hat          [K, N, 16]            prediction vectors  û_{j|i} = W_ij u_i
        iterations     per routing iteration: c [K, N], b [K, N], s [K, 16],
                       v [K, 16], agreement [K, N] (û_{j|i} · v_j)
        digit_norms    [n_iter, K]
        predicted      int
        reconstruction display image from the decoder
    """
    geo = geometry_of(model)
    conv1 = F.relu(model.conv1(x))
    u = model.primary(conv1)
    v_final, _, _, history = model.digits(u)
    W = model.digits.routing_module.W
    u_hat = torch.matmul(W, u.unsqueeze(1).unsqueeze(-1)).squeeze(-1)[0]

    iterations = []
    for h in history:
        c = h["coupling_coeffs"][0]
        v = h["digit_capsules"][0]
        s = (c.unsqueeze(-1) * u_hat).sum(dim=1)
        iterations.append({
            "c": c.cpu().numpy(),
            "b": h["agreement_scores"][0].cpu().numpy(),
            "s": s.cpu().numpy(),
            "v": v.cpu().numpy(),
            "agreement": (u_hat * v.unsqueeze(1)).sum(dim=-1).cpu().numpy(),
        })

    # Reconstruction exactly as trained: decoder input masked to the predicted class.
    # (CIFAR-10's decoder was trained against normalized pixels through a sigmoid and
    # cannot represent them, so its output is not shown; see spec.mean.)
    recon = model.reconstruct(v_final).view(spec.in_channels, spec.img_size, spec.img_size)
    recon = None if spec.mean is not None else recon[0].clamp(0, 1).cpu().numpy()

    digit_norms = np.stack([np.linalg.norm(it["v"], axis=-1) for it in iterations])
    return {
        "geo": geo,
        "image": spec.to_display(x[0]),
        "conv1": conv1[0].cpu().numpy(),
        "u": u[0].cpu().numpy(),
        "u_norm": torch.norm(u[0], dim=-1).cpu().numpy(),
        "u_hat": u_hat.cpu().numpy(),
        "iterations": iterations,
        "digit_norms": digit_norms,
        "predicted": int(np.argmax(digit_norms[-1])),
        "reconstruction": recon,
    }


def attributions(model, x):
    """Exact ablation, gradient attributions and baselines for x (see attribution.py)."""
    return AT.explain(model, x)


def squash_np(s, eps=1e-8):
    """numpy twin of capsule_core.squash, for the squash chapter."""
    s = np.asarray(s, dtype=float)
    sq = np.sum(s * s, axis=-1, keepdims=True)
    return sq / (1.0 + sq) * s / np.sqrt(sq + eps)


__all__ = ["get_device", "pick_index", "trace_routing", "attributions", "squash_np", "squash"]
