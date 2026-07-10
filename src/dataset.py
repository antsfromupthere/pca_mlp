from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class RFDataset(Dataset):
    def __init__(self, x, power, target_z):
        self.x = torch.from_numpy(x).float()
        self.power = torch.from_numpy(power).float()
        self.target = torch.from_numpy(target_z).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.power[idx], self.target[idx]


def make_features(features , label,  n_STATE, stats=None):
    """
    Build features = [normalized xy, state one-hot].
    On the training call, pass stats=None and the function returns (features, stats).
    On val/test, pass the stats returned from the training call.
    """
    if stats is None:
        xy_mean = features.mean(axis=0, keepdims=True)
        xy_std = features.std(axis=0, keepdims=True)
        xy_std[xy_std < 1e-6] = 1.0
        xy_min = features.min(axis=0, keepdims=True)
        xy_range = features.max(axis=0, keepdims=True) - xy_min
        xy_range[xy_range < 1e-6] = 1.0
        stats = {
            "xy_mean": xy_mean.astype(np.float32),
            "xy_std": xy_std.astype(np.float32),
            "xy_min": xy_min.astype(np.float32),
            "xy_range": xy_range.astype(np.float32),
        }

    xy_norm = ((features - stats["xy_mean"]) / stats["xy_std"]).astype(np.float32)
    parts = [xy_norm]

    parts.append(np.eye(n_STATE, dtype=np.float32)[label])
    return np.concatenate(parts, axis=1).astype(np.float32), stats


