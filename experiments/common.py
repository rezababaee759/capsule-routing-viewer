"""Shared pieces for the experiments: image sets, determinism and provenance (see PROTOCOL.md)."""

import datetime
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explorer import protocol as P                                           # noqa: E402

RESULTS = ROOT / "results"
SOURCE_DIRS = ("explorer", "experiments", "scripts", "tests")


def set_determinism(seed=P.SEED):
    """Deterministic kernels where PyTorch offers them (results are compared value-level)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def pilot_indices(n_test):
    """The pilot images that preceded the protocol; excluded from every evaluation set."""
    return set(np.random.default_rng(P.PILOT_SEED).choice(n_test, P.PILOT_N, replace=False).tolist())


def main_set(targets, n_per_class=P.N_PER_CLASS, seed=P.SEED):
    """Per class, a seeded random draw of n_per_class test images (pilot excluded).
    Returned class-major: all images of class 0, then class 1, ..."""
    targets = np.asarray(targets)
    pilot = pilot_indices(len(targets))
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(targets):
        pool = np.array([i for i in np.flatnonzero(targets == c) if i not in pilot])
        out.extend(rng.permutation(pool)[:n_per_class].tolist())
    return out


@torch.no_grad()
def predictions(model, ds, device, batch=500):
    preds = []
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=batch):
        v, *_ = model(x.to(device))
        preds.append(v.norm(dim=-1).argmax(1).cpu())
    return torch.cat(preds).numpy()


def misclassified_set(model, ds, targets, device, exclude, n=P.MISCLASSIFIED_MAX,
                      seed=P.SEED + P.MISCLASSIFIED_SEED_OFFSET):
    """Up to n test errors of `model`, seeded random order, excluding `exclude`.
    Also returns the full test-set accuracy of the model."""
    pred = predictions(model, ds, device)
    wrong = [i for i in np.flatnonzero(pred != np.asarray(targets)) if i not in exclude]
    rng = np.random.default_rng(seed)
    return sorted(rng.permutation(wrong)[:n].tolist()), float((pred == np.asarray(targets)).mean())


def additivity_set(n_test, exclude):
    """Every test image outside `exclude` (pilot, main set, misclassified set), in index order."""
    return [i for i in range(n_test) if i not in exclude]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def code_fingerprint():
    """md5 over every source file (path + content) in the analysis code. Recorded in
    every artifact so a result can be tied to the exact code that produced it."""
    h = hashlib.md5()
    files = sorted(p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*")
                   if p.suffix in (".py", ".js", ".md") and "__pycache__" not in p.parts)
    files += [ROOT / "capsnet_routing_explorer.py", ROOT / "capsule_core.py"]
    for p in files:
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                                # noqa: BLE001
        return None


def versions():
    import scipy
    out = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "scipy": scipy.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None}
    for mod in ("matplotlib", "plotly", "streamlit", "torchvision"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                            # noqa: BLE001
            pass
    return out


def provenance(**extra):
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "code_fingerprint": code_fingerprint(),
        "git_commit": git_commit(),
        "versions": versions(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "protocol": "experiments/PROTOCOL.md; constants in explorer/protocol.py",
        **extra,
    }
