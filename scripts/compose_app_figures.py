#!/usr/bin/env python3
"""
Composes the raw app capture (scripts/capture_app.py) into the paper figure
fig_hero: the full app view (cube with its overlays) for MNIST.
"""

from matplotlib import pyplot as plt
from PIL import Image

from figstyle import FIG, FULL, save

RAW = FIG / "raw"


def fig_hero():
    fig, ax = plt.subplots(figsize=(FULL, FULL * 0.62))
    ax.imshow(Image.open(RAW / "hero_mnist.png"))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "fig_hero")


if __name__ == "__main__":
    fig_hero()
