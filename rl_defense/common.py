"""Shared paths, data helpers and model factories for the RL defence.

The original repository code is imported, never modified.
"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.benchmark = True

ROOT = Path(__file__).resolve().parent.parent          # .../URVFL
CIFAR_DIR = ROOT / "cifar10"
for p in (str(ROOT), str(CIFAR_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from torchvision import datasets, transforms                     # noqa: E402
from torch.utils.data import DataLoader, Subset                  # noqa: E402
from data_dis import Distribute_MNIST                            # noqa: E402  (original code)
import models as M                                               # noqa: E402  (original code)

RL_DIR = ROOT / "rl_defense"
CACHE_DIR = RL_DIR / "cache"
RESULTS_DIR = ROOT / "cifar10" / "rl_defense_result"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = os.environ.get("URVFL_DATA_ROOT", str(ROOT.parent / "data"))
OWNERS = ["client_1", "client_2"]


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def base_config():
    """The original URVFL config, untouched, plus the RL defence config."""
    cfg = load_json(CIFAR_DIR / "config_cifar.json")
    rl = load_json(CIFAR_DIR / "config_rl_defense.json")
    return cfg, rl


TRANSFORM_TRAIN = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
TRANSFORM_TEST = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def cifar_train(transform=TRANSFORM_TRAIN):
    return datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)


def cifar_test():
    return datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=TRANSFORM_TEST)


def distributed(dataset, batch_size, shuffle=True, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return Distribute_MNIST(data_owners=OWNERS, data_loader=loader, device=device)


def simulation_split(n_victim=10000, n_aux_pool=6000, seed=1234):
    """Fixed index split of the CIFAR-10 train set used only by the offline
    simulator (a stand-in for public surrogate data the defender designer has)."""
    g = np.random.RandomState(seed)
    perm = g.permutation(50000)
    return perm[:n_victim], perm[n_victim:n_victim + n_aux_pool]


def aux_indices(pool_idx, targets, kind, rng):
    """Attacker auxiliary-data distributions used to vary the simulated attacks."""
    pool_idx = np.asarray(pool_idx)
    labels = np.asarray(targets)[pool_idx]
    if kind == "uniform":
        return rng.choice(pool_idx, 4545, replace=False)
    if kind == "small":
        return rng.choice(pool_idx, 1000, replace=False)
    if kind == "skewed":                       # 80% of aux from 3 classes
        major = rng.choice(10, 3, replace=False)
        m = np.isin(labels, major)
        a = rng.choice(pool_idx[m], min(2400, m.sum()), replace=False)
        b = rng.choice(pool_idx[~m], 600, replace=False)
        return np.concatenate([a, b])
    if kind == "half_classes":                 # aux only covers 5 of the 10 classes
        cls = rng.choice(10, 5, replace=False)
        m = np.isin(labels, cls)
        return rng.choice(pool_idx[m], min(2500, m.sum()), replace=False)
    raise ValueError(kind)


def client_net():
    return M.ClientNet(34, 10)


class ReferenceAE(torch.nn.Module):
    """Independent reference encoder (+ local decoder used only for its training)."""

    def __init__(self):
        super().__init__()
        self.encoder = client_net()
        self.decoder = M.fullDecoder()

    def forward(self, x):
        return self.encoder(x)
