"""
Routing analysis helpers.

All spatial lookups go through a registry.Geometry (G x G grid, T types), whose
flattening order matches PrimaryCapsules.forward:  index = (gy * G + gx) * T + t.
"""

import numpy as np
from scipy.stats import spearmanr

from .protocol import TOPK_FRACS

EPS = 1e-12


def calculate_entropy(coupling_coeffs):
    """Routing entropy (bits) of each primary capsule's coupling distribution. [N]"""
    c = np.clip(coupling_coeffs, EPS, 1.0)
    return -np.sum(c * np.log2(c), axis=0)


def normalized_entropy(coupling_coeffs):
    K = coupling_coeffs.shape[0]
    return np.clip(calculate_entropy(coupling_coeffs) / np.log2(K), 0.0, 1.0)


def grid_sum(geo, values):
    """Total over the T capsule types at each grid cell. [G (gy), G (gx)]"""
    return geo.to_grid(values).sum(axis=0)


def topk_overlap(scores, reference, frac=TOPK_FRACS[0]):
    k = max(1, int(np.ceil(frac * len(reference))))
    return len(set(np.argsort(-scores)[:k]) & set(np.argsort(-reference)[:k])) / k


def agreement(scores, reference):
    """How well an attribution recovers exact ablation, for the UI readout."""
    return {"rho": float(spearmanr(scores, reference)[0]),
            "p1": topk_overlap(scores, reference, TOPK_FRACS[0]),
            "p5": topk_overlap(scores, reference, TOPK_FRACS[1])}


def rank_of(values, i):
    """1-based rank of element i when sorted descending."""
    return int(np.sum(values > values[i])) + 1


def get_capsule_info(geo, cap_idx, coupling_coeffs, primary_norms, primary_vectors=None):
    gx, gy, t = geo.coords(cap_idx)
    info = {
        "cap_idx": int(cap_idx),
        "grid_position": (gx, gy),
        "capsule_type": t,
        "activation_norm": float(primary_norms[cap_idx]),
        "votes": coupling_coeffs[:, cap_idx],
        "receptive_field": geo.receptive_field(gx, gy),
    }
    if primary_vectors is not None:
        info["activation_vector"] = primary_vectors[cap_idx]
    return info
