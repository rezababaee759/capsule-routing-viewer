"""
Checks the statistics code in experiments/summarize.py against reference
implementations. Run with pytest, or directly: python tests/test_stats.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

import summarize as S                                                       # noqa: E402


def test_holm_matches_statsmodels():
    p = np.random.default_rng(0).uniform(0, 0.2, 9)
    assert np.allclose(S.holm(p), multipletests(p, method="holm")[1])


def test_rank_biserial_matches_wilcoxon_statistic():
    d = np.random.default_rng(1).normal(0.3, 1, 200)
    W_plus = wilcoxon(d, alternative="greater").statistic        # sum of ranks of positive d
    total = len(d) * (len(d) + 1) / 2
    assert abs(S.rank_biserial(d) - (2 * W_plus / total - 1)) < 1e-9
    assert S.rank_biserial(np.abs(d) + 0.1) == 1.0 and S.rank_biserial(np.zeros(5)) == 0.0


def test_bootstrap_mean_ci_matches_normal_theory():
    rng = np.random.default_rng(2)
    x = rng.normal(5, 2, 500)
    strata = np.repeat(np.arange(10), 50)
    theta, (lo, hi), method = S.ci(x, strata, stat=np.mean)
    se = 2 / np.sqrt(500)
    assert method == "percentile"
    assert abs((hi - lo) - 2 * 1.96 * se) / (2 * 1.96 * se) < 0.1     # width within 10%
    assert lo < 5 < hi


def test_degenerate_and_bca_paths():
    strata = np.repeat(np.arange(10), 50)
    _, ci_, m = S.ci(np.ones(500), strata)
    assert m == "degenerate" and ci_ == [1.0, 1.0]
    x = np.random.default_rng(3).exponential(1, 500) ** 2          # strongly skewed
    theta, (lo, hi), m = S.ci(x, strata, stat=np.mean)
    assert m == "BCa" and lo < theta < hi


if __name__ == "__main__":
    failed = 0
    for name, fn in [(n, f) for n, f in globals().items() if n.startswith("test_")]:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as e:                                          # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
