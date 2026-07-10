import torch

def add_coordinate_jitter(x_batch, sigma):
    x_aug = x_batch.clone()
    noise = sigma * torch.randn_like(x_aug[:, :2])
    x_aug[:, :2] = x_aug[:, :2] + noise
    return x_aug