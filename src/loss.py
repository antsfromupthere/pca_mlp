import torch.nn.functional as F


def pca_coeff_loss(pred_coeff, target_coeff):
    #  MSE loss on PCA coefficients.#
    return F.mse_loss(pred_coeff, target_coeff)