import numpy as np


def fit_pca_decoder_targets(
    label_train,
    label_val,
    label_test,
    n_components,
    min_gap_db,
):
    # fit PCA on the training scaled-gap matrix
    train_scaled_gap = make_scaled_gap_np(label_train, min_gap_db=min_gap_db)
    val_scaled_gap   = make_scaled_gap_np(label_val,   min_gap_db=min_gap_db)
    test_scaled_gap  = make_scaled_gap_np(label_test,  min_gap_db=min_gap_db)

    # mean over training samples
    mean_scaled_gap = train_scaled_gap.mean(axis=0).astype(np.float32)
    train_centered = train_scaled_gap - mean_scaled_gap[None, :]

    # SVD
    U, S, Vt = np.linalg.svd(train_centered, full_matrices=False)

    var = S ** 2
    ratio = var / np.maximum(var.sum(), 1e-12)
    cum = np.cumsum(ratio)

    c_components = min(n_components, Vt.shape[0])
    components = Vt[:c_components+1].astype(np.float32)   # (r, N_cfg), this is V_r^T

    def project(scaled_gap):
        centered = scaled_gap - mean_scaled_gap[None, :]
        return (centered @ components.T).astype(np.float32)

    train_coeff = project(train_scaled_gap)
    val_coeff   = project(val_scaled_gap)
    test_coeff  = project(test_scaled_gap)

    # per-component std
    coeff_std = train_coeff.std(axis=0).astype(np.float32)

    pca_info = {
        "mean_scaled_gap": mean_scaled_gap,
        "components": components,
        "singular_values": S.astype(np.float32),
        "explained_variance_ratio": ratio.astype(np.float32),
        "explained_variance_cum": cum.astype(np.float32),
        "coeff_std": coeff_std,
        "train_coeff": train_coeff,
        "val_coeff": val_coeff,
        "test_coeff": test_coeff,
    }

    return pca_info


def make_scaled_gap_np(power, min_gap_db):
    gap = power - power.max(axis=1, keepdims=True)
    gap = np.clip(gap, min_gap_db, 0.0)
    scaled_gap = gap / abs(min_gap_db)
    return scaled_gap.astype(np.float32)


def reconstruct_scaled_gap_from_coeff(coeff, pca_info):
    """ reconstruction: scaled_gap_hat = mu + c @ V_r^T"""
    return (
        pca_info["mean_scaled_gap"][None, :]
        + coeff @ pca_info["components"]
    ).astype(np.float32)


def reconstruct_scaled_gap_from_coeff_torch(coeff, mean_scaled_gap, components):

    return mean_scaled_gap.unsqueeze(0) + coeff @ components