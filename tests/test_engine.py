"""
Correctness tests for the attribution engine.

Run with pytest, or directly:  python tests/test_engine.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capsule_core import squash                                   # noqa: E402
from explorer import attribution as AT                            # noqa: E402
from explorer.data import squash_np                               # noqa: E402
from explorer.registry import DATASETS, geometry_of               # noqa: E402
import run_additivity as RA                                       # noqa: E402
import run_margin as RM                                           # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CACHE = {}


def _setup(key):
    if key not in _CACHE:
        spec = DATASETS[key]
        model = spec.build_model(DEV)
        x = spec.dataset()[3][0].unsqueeze(0).to(DEV)
        _CACHE[key] = (spec, model, x, AT.primary_capsules(model, x))
    return _CACHE[key]


def test_index_convention():
    """cap_index(gx, gy, t) must address the capsule built from conv channel group t at (gy, gx)."""
    for key in ("mnist", "cifar10"):
        spec, model, x, u = _setup(key)
        geo = geometry_of(model)
        with torch.no_grad():
            raw = model.primary.conv(torch.relu(model.conv1(x)))           # [1, T*8, G, G]
            raw = raw.view(1, geo.T, 8, geo.G, geo.G)
        rng = np.random.default_rng(0)
        for _ in range(20):
            gx, gy, t = rng.integers(geo.G), rng.integers(geo.G), rng.integers(geo.T)
            expected = squash(raw[0, t, :, gy, gx].unsqueeze(0))[0]
            assert torch.allclose(u[0, geo.index(gx, gy, t)], expected, atol=1e-5)
            assert geo.coords(geo.index(gx, gy, t)) == (gx, gy, t)


def test_batched_ablation_matches_loop():
    for key in ("mnist", "cifar10"):
        spec, model, x, u = _setup(key)
        j = AT.predicted_class(model, u)
        batched = AT.exact_ablation(model, u, j, chunk=97)                # odd chunk on purpose
        with torch.no_grad():
            phi0 = AT.confidence(model.digits(u)[0], j)[0].item()
            for i in np.random.default_rng(1).choice(u.shape[1], 12, replace=False):
                ua = u.clone(); ua[0, i] = 0
                ref = phi0 - AT.confidence(model.digits(ua)[0], j)[0].item()
                assert abs(batched[i] - ref) < 1e-6, (key, i, batched[i], ref)


def test_gradients_match_finite_differences():
    """Directional derivative of φ along random capsule directions, in float64."""
    spec = DATASETS["mnist"]
    model = spec.build_model(torch.device("cpu")).double()
    x = spec.dataset()[3][0].unsqueeze(0).double()
    u = AT.primary_capsules(model, x)
    grads, j, _ = AT.gradient_attributions(model, u)
    u_req = u.clone().requires_grad_(True)
    v, *_ = model.digits(u_req)
    g, = torch.autograd.grad(AT.confidence(v, j)[0], u_req)
    rng = np.random.default_rng(2)
    eps = 1e-6
    for _ in range(5):
        d = torch.as_tensor(rng.normal(size=u.shape))
        with torch.no_grad():
            fp = AT.confidence(model.digits(u + eps * d)[0], j)[0]
            fm = AT.confidence(model.digits(u - eps * d)[0], j)[0]
        fd = (fp - fm) / (2 * eps)
        assert abs(float(fd) - float((g * d).sum())) < 1e-6 * max(1.0, abs(float(fd)))
    # taylor is <g_i, u_i>
    assert np.allclose(grads["taylor"], (g[0] * u[0]).sum(-1).numpy(), atol=1e-8)


def test_squash_and_softmax():
    s = np.random.default_rng(3).normal(size=(10, 16))
    assert np.allclose(squash_np(s), squash(torch.as_tensor(s)).numpy(), atol=1e-6)
    spec, model, x, u = _setup("mnist")
    with torch.no_grad():
        _, c, _, _ = model.digits(u)
    assert torch.allclose(c.sum(dim=1), torch.ones_like(c.sum(dim=1)), atol=1e-5)


def test_curve_endpoints():
    spec, model, x, u = _setup("mnist")
    j = AT.predicted_class(model, u)
    scores = np.random.default_rng(4).random(u.shape[1])
    frac, dele, ins = AT.deletion_insertion(model, u, j, scores, steps=10)
    assert frac[0] == 0 and frac[-1] == 1
    assert abs(dele[0] - 1) < 1e-6 and abs(ins[-1] - 1) < 1e-6
    assert dele[-1] < 1e-6 and ins[0] < 1e-6


def test_frozen_deletion_matches_masked_passes():
    """Closed-form frozen curve and Δ^F against explicit passes with the intact couplings fixed."""
    spec, model, x, u = _setup("mnist")
    j = AT.predicted_class(model, u)
    w = AT.frozen_votes(model, u, j)
    with torch.no_grad():
        v, c, _, _ = model.digits(u)
        W = model.digits.routing_module.W

        def phi_frozen(keep):
            u_hat = torch.matmul(W, (u * keep.unsqueeze(-1)).unsqueeze(1).unsqueeze(-1)).squeeze(-1)
            return float(squash((c.unsqueeze(-1) * u_hat).sum(dim=2))[0, j].norm())

    order = np.random.default_rng(5).permutation(u.shape[1])
    _, phi_f, cos_f = AT.frozen_deletion(w, order)
    assert abs(phi_f[0] - float(AT.confidence(v, j)[0])) < 1e-6
    assert phi_f[-1] == 0 and abs(cos_f[0] - 1) < 1e-12 and np.isnan(cos_f[-1])
    exf = AT.frozen_ablation(w)
    for k in (1, 7, 100, 600, u.shape[1] - 1):
        keep = torch.ones(1, u.shape[1], device=u.device)
        keep[0, torch.as_tensor(order[:k])] = 0
        assert abs(phi_f[k] - phi_frozen(keep)) < 1e-6, k
    for i in order[:5]:
        keep = torch.ones(1, u.shape[1], device=u.device)
        keep[0, i] = 0
        assert abs(exf[i] - (phi_f[0] - phi_frozen(keep))) < 1e-6, i


def test_routed_deletion_curve():
    """φ_R(0) = φ0, φ_R(N) = 0, the first removal equals the top single effect, and the
    curve does not depend on how the masks are batched."""
    spec, model, x, u = _setup("mnist")
    j = AT.predicted_class(model, u)
    ex, order, phi_r = RA.image_core(model, u, j)
    with torch.no_grad():
        phi0 = float(AT.confidence(model.digits(u)[0], j)[0])
    assert abs(phi_r[0] - phi0) < 1e-6 and phi_r[-1] < 1e-6
    assert abs((phi_r[0] - phi_r[1]) - ex[order[0]]) < 1e-6
    ks = np.array([0, 3, 50, 400, u.shape[1]])
    assert np.allclose(RA.routed_deletion(model, u, j, order, ks, chunk=7), phi_r[ks], atol=1e-6)


def test_additive_readout_has_no_pre_negative_rebound():
    """An exactly additive readout (s_ĵ under fixed couplings, projected on its intact
    direction) dips and recovers once negative effects are removed, but never rises
    while only positive effects are being removed."""
    spec, model, x, u = _setup("mnist")
    j = AT.predicted_class(model, u)
    w = AT.frozen_votes(model, u, j).double().cpu().numpy()
    e = w.sum(0) / np.linalg.norm(w.sum(0))
    eff = w @ e
    order = np.argsort(-eff, kind="stable")
    y = (w.sum(0) - np.concatenate([np.zeros((1, w.shape[1])), np.cumsum(w[order], 0)])) @ e
    y = y / y[0]
    kplus = int((eff > 0).sum())
    assert np.allclose(1 - y, np.concatenate([[0], np.cumsum(eff[order])]) / (w.sum(0) @ e))
    assert RA.drawup(y, kplus) < 1e-12
    assert RA.drawup(y, len(eff) - 1) > 0


def test_tolerance_markers():
    ex = np.array([0.5, 0.2, 1e-9, -1e-9, 0.0, -0.1, -0.3])
    order = np.argsort(-ex, kind="stable")
    y = np.linspace(1, 0, len(ex) + 1)
    t = RA.thresholded(ex, ex, order, y, y, y, 1e-6)
    assert (t["kplus"], t["kminus"], t["n_null"]) == (2, 6, 3)
    assert t["pre_rise"] == 0
    assert abs(RA.drawup(np.array([1, .5, .8, .2, .6]), 4) - 0.4) < 1e-12


def test_margin_readout():
    """The default readout is φ; the margin readout is ‖v_j‖ minus the longest other class capsule,
    its batched single ablations match explicit masked passes, and the Taylor margin estimate is
    the derivative along α u_i of ‖v_j‖ − ‖v_r‖ with the intact runner-up r."""
    spec, model, x, u = _setup("mnist")
    j = AT.predicted_class(model, u)
    keep = (torch.rand(9, u.shape[1], generator=torch.Generator().manual_seed(3)) > 0.3).to(DEV)
    assert torch.equal(AT.phi_of_masks(model, u, keep, j), AT.phi_of_masks(model, u, keep, j, readout=AT.confidence))
    with torch.no_grad():
        v, *_ = model.digits(u.expand(9, -1, -1) * keep.unsqueeze(-1))
        lengths = v.norm(dim=-1)
        runner = torch.cat([lengths[:, :j], lengths[:, j + 1:]], dim=1).max(dim=1).values
        assert torch.allclose(AT.phi_of_masks(model, u, keep, j, readout=AT.margin), lengths[:, j] - runner)
        intact, abl = RM.single_ablations(model, u, j, chunk=61)
        for i in (0, 17, 500):
            ua = u.clone(); ua[0, i] = 0
            ref = RM.readouts(model.digits(ua)[0], j)[0]
            assert torch.allclose(abl[i], ref, atol=1e-6)
    model64 = spec.build_model(torch.device("cpu")).double()
    u64 = AT.primary_capsules(model64, x.cpu().double())
    with torch.no_grad():
        lengths = model64.digits(u64)[0][0].norm(dim=-1)
    others = lengths.clone(); others[j] = -torch.inf
    r = int(others.argmax())
    tm = RM.taylor_margin(model64, u64, j, r)
    eps = 1e-6
    for i in (0, 17, 500):
        def m(a):
            ua = u64.clone(); ua[0, i] = a * u64[0, i]
            with torch.no_grad():
                L = model64.digits(ua)[0][0].norm(dim=-1)
            return float(L[j] - L[r])
        assert abs((m(1 + eps) - m(1 - eps)) / (2 * eps) - tm[i]) < 1e-6 * max(1.0, abs(tm[i]))


if __name__ == "__main__":
    failed = 0
    for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:                                          # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
