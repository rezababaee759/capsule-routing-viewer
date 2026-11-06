"""
Print style for the paper's figures.

Palette: checked with the dataviz skill's palette validator (light mode, white
surface): the five family hues pass its adjacent-pair checks and contrast check, and
the first three pass its all-pairs check (used for scatter / small multiples).
Methods are coloured by FAMILY, trivial baselines are neutral grey + dashed, and the
exact-ablation reference is ink.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib import font_manager as fm                                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette
INK = "#1a1a1a"
INK2 = "#55524d"          # secondary text
MUTED = "#8c8a85"
GRID = "#e6e4df"
PAPER = "#ffffff"
BLUE, GOLD, TEAL, PURPLE, RED = "#2b6cb0", "#c06d1a", "#14917a", "#8a4fb0", "#c2414b"
GREY1, GREY2 = "#6b6b6b", "#a8a6a1"

# app (dark, 3Blue1Brown) colours, for captions that refer to the screenshots
APP = {"blue": "#58C4DD", "gold": "#F0AC5F", "teal": "#5CD0B3", "red": "#FC6255", "yellow": "#FFFF00"}

METHOD = {   # key: (label, colour, family, linestyle, marker)
    "exact":          ("exact ablation",               INK,    "reference", "--", "o"),
    "taylor":         (r"Taylor, through routing",     BLUE,   "taylor",    "-",  "o"),
    "taylor_frozen":  (r"Taylor, routing frozen",      GOLD,   "frozen",    "-",  "s"),
    "g_C":            (r"$|\partial\varphi/\partial c_{i\hat\jmath}|$", TEAL, "magnitude", "-", "D"),
    "g_A":            (r"$\|\partial\varphi/\partial u_i\|$",           TEAL, "magnitude", "-.", "^"),
    "coupling":       (r"coupling $c_{i\hat\jmath}$",  RED,    "state",     "-",  "X"),
    "act_x_coupling": (r"$\|u_i\|\,c_{i\hat\jmath}$",  RED,    "state",     "--", "x"),
    "activation":     (r"activation $\|u_i\|$",        GREY1,  "trivial",   "--", "."),
    "random":         ("random",                       GREY2,  "trivial",   ":",  "."),
}
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explorer.registry import DATASETS as _DATASETS                          # noqa: E402
DATASET_TITLE = {k: v.title for k, v in _DATASETS.items()}

# ---------------------------------------------------------------- typography
# Pin the 10-pt optical master of Latin Modern (Computer Modern look, as in the app).
fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist
                          if "lmroman" not in f.fname.lower()
                          or "lmroman10-" in f.fname.lower()]
if not any("lmroman10-" in f.fname.lower() for f in fm.fontManager.ttflist):
    raise RuntimeError("Latin Modern Roman 10 not found: install lmodern / texlive-fonts so figures match")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round",
    "figure.dpi": 150,
    "savefig.dpi": 600,          # rasterised layers inside the PDFs
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
})

# MDPI text block is ~16.9 cm wide
FULL = 16.9 / 2.54
HALF = 8.2 / 2.54


def save(fig, name):
    # No creation timestamps, so re-running a script reproduces the file byte-for-byte.
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=600, metadata={"Software": None})   # >= 300 dpi at print width
    plt.close(fig)
    print(f"  saved paper/figures/{name}.pdf/.png")


def panel_label(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="right", va="bottom")


def case_index(dataset, case):
    """Rule-selected case-study image (scripts/figure_data.py) -- never typed by hand."""
    import json as _json
    import numpy as _np
    return int(_json.loads(str(_np.load(ROOT / "results" / "cache" / f"figdata_{dataset}.npz")["cases"]))[case])
