"""
Dataset / checkpoint registry.

One entry per dataset: how to build the CapsNet that matches its checkpoints,
the input-transform contract the checkpoints were trained under, and how to turn
a model input back into a displayable image. Capsule-grid geometry is derived
from the model, never stored.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from capsule_core import CapsuleNetwork
from explorer.protocol import REFERENCE_VARIANT

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

# Input-transform contract of the CIFAR-10 checkpoints: these are the constants used by
# the training script that produced them (CapsNet-App training/train_cifar10.py, also
# core/transforms.py there). They are the widely copied CIFAR-10 normalisation values,
# not the exact per-channel std of the training set; what matters here is that they
# match training, so do not "correct" them.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Training configuration of every checkpoint (not stored in the state dict):
# dynamic routing with 3 iterations (Sabour et al., 2017).
ROUTING_ITERS = 3

# Margin-loss constants used in training (Sabour et al., 2017): m+ = 0.9, m- = 0.1, lambda = 0.5.
M_PLUS, M_MINUS, LAMBDA = 0.9, 0.1, 0.5


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    in_channels: int
    img_size: int
    n_types: int
    classes: tuple
    mean: tuple = field(default=None)
    std: tuple = field(default=None)

    # ---- model -------------------------------------------------------------
    def checkpoint(self, variant=None):
        """Checkpoint file of a training run; by default the dataset's reference run."""
        return MODELS / f"capsule_{self.key}_dynamic_{variant or REFERENCE_VARIANT[self.key]}.pt"

    def build_model(self, device, variant=None):
        model = CapsuleNetwork(in_channels=self.in_channels, num_classes=len(self.classes),
                               routing_iters=ROUTING_ITERS,
                               img_width=self.img_size, img_height=self.img_size,
                               num_primary_capsules=self.n_types).to(device)
        state = torch.load(self.checkpoint(variant), map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model

    # ---- data --------------------------------------------------------------
    def transform(self):
        t = [transforms.ToTensor()]
        if self.mean is not None:
            t.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(t)

    def dataset(self, train=False):
        cls = {"mnist": datasets.MNIST, "fashionmnist": datasets.FashionMNIST,
               "cifar10": datasets.CIFAR10}[self.key]
        return cls(root=str(DATA), train=train, download=False, transform=self.transform())

    def targets(self, ds):
        return np.asarray(ds.targets)

    def to_display(self, x):
        """Model input [C, H, W] tensor -> displayable [H, W] or [H, W, 3] in [0, 1]."""
        x = x.detach().cpu().float()
        if self.mean is not None:
            x = x * torch.tensor(self.std)[:, None, None] + torch.tensor(self.mean)[:, None, None]
        x = x.clamp(0, 1).numpy()
        return x[0] if x.shape[0] == 1 else x.transpose(1, 2, 0)


DATASETS = {
    "mnist": DatasetSpec("mnist", "MNIST", 1, 28, 32, tuple(str(d) for d in range(10))),
    "fashionmnist": DatasetSpec(
        "fashionmnist", "Fashion-MNIST", 1, 28, 32,
        ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")),
    "cifar10": DatasetSpec(
        "cifar10", "CIFAR-10", 3, 32, 64,
        ("airplane", "automobile", "bird", "cat", "deer",
         "dog", "frog", "horse", "ship", "truck"),
        mean=CIFAR10_MEAN, std=CIFAR10_STD),
}


@dataclass(frozen=True)
class Geometry:
    """Primary-capsule grid and the architecture facts the app and paper print.
    G x G spatial positions, T types, N = G*G*T capsules; flattening order
    (PrimaryCapsules.forward): index = (gy * G + gx) * T + t. Every field is read
    from the model by geometry_of()."""
    G: int
    T: int
    img_size: int
    k1: int          # conv1 kernel
    s1: int          # conv1 stride
    k2: int          # primary-capsule conv kernel
    s2: int          # primary-capsule conv stride
    conv1_ch: int    # conv1 output channels
    d_in: int        # primary capsule dimension
    d_out: int       # class capsule dimension
    K: int           # number of class capsules

    @property
    def N(self):
        return self.G * self.G * self.T

    @property
    def rf(self):
        """Receptive field (pixels) of one primary capsule: k1 + (k2 - 1) * s1."""
        return self.k1 + (self.k2 - 1) * self.s1

    @property
    def stride(self):
        """Input-pixel step between neighbouring grid cells: s1 * s2."""
        return self.s1 * self.s2

    def index(self, gx, gy, t):
        return (gy * self.G + gx) * self.T + t

    def coords(self, idx):
        spatial, t = divmod(int(idx), self.T)
        gy, gx = divmod(spatial, self.G)
        return gx, gy, t

    def to_grid(self, values):
        """[N] -> [T, G (gy), G (gx)]"""
        return np.asarray(values).reshape(self.G, self.G, self.T).transpose(2, 0, 1)

    def receptive_field(self, gx, gy):
        """Input window of grid cell (gx, gy): (x0, x1, y0, y1), half-open, clipped."""
        clip = lambda v: max(0, min(self.img_size, v))
        x0, y0 = self.stride * gx, self.stride * gy
        return clip(x0), clip(x0 + self.rf), clip(y0), clip(y0 + self.rf)

    def rf_center(self, gx, gy):
        x0, _, y0, _ = self.receptive_field(gx, gy)
        return x0 + (self.rf - 1) / 2, y0 + (self.rf - 1) / 2


def geometry_of(model):
    """Derive the capsule grid and layer facts from the model itself."""
    c1, pc = model.conv1, model.primary.conv
    img = model.img_width
    g1 = (img - c1.kernel_size[0]) // c1.stride[0] + 1
    G = (g1 - pc.kernel_size[0]) // pc.stride[0] + 1
    W = model.digits.routing_module.W                     # [1, K, N, d_out, d_in]
    return Geometry(G=G, T=model.primary.num_capsules, img_size=img,
                    k1=c1.kernel_size[0], s1=c1.stride[0], k2=pc.kernel_size[0], s2=pc.stride[0],
                    conv1_ch=c1.out_channels, d_in=W.shape[4], d_out=W.shape[3], K=W.shape[1])
