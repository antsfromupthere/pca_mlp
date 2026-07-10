import torch
import numpy as np

from src.loss import pca_coeff_loss


@torch.no_grad()
def evaluate_model(model, loader, k_list, device):
    """
    Evaluate the PCA-decoder model.

    Tracks:
      - mean coefficient MSE (for sanity-checking train/val gap)
      - regret / hit / within-X-dB metrics for top-K beam selection

    Top-K selection uses `pred_scaled_gap` (the reconstruction produced
    by the fixed PCA decoder); larger values correspond to configurations
    predicted to be closer to the optimum.
    """
    model.eval()

    max_k = max(k_list)

    total_loss = 0.0
    total_num = 0

    regrets = {k: [] for k in k_list}
    hits = {k: [] for k in k_list}

    for x, power, target_coeff in loader:
        x = x.to(device)
        power = power.to(device)
        target_coeff = target_coeff.to(device)

        pred_coeff, pred_scaled_gap = model(x)

        loss = pca_coeff_loss(pred_coeff, target_coeff)

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_num += bs

        true_best_power, true_best_idx = power.max(dim=1)
        pred_topk_idx = torch.topk(pred_scaled_gap, k=max_k, dim=1).indices

        for k in k_list:
            idx_k = pred_topk_idx[:, :k]
            selected_power = power.gather(1, idx_k).max(dim=1).values
            regret = true_best_power - selected_power
            hit = (idx_k == true_best_idx.unsqueeze(1)).any(dim=1).float()

            regrets[k].extend(regret.detach().cpu().numpy().astype(float).tolist())
            hits[k].extend(hit.detach().cpu().numpy().astype(float).tolist())

    metrics = compute_metrics_from_regrets_and_hits(regrets, hits, k_list)
    metrics["loss"] = total_loss / total_num
    metrics["coeff_loss"] = metrics["loss"]  # alias, in case other code references it

    return metrics


def format_metrics(prefix, metrics, k_list):
    msg = f"{prefix} loss={metrics['loss']:.6f}"
    for k in k_list:
        msg += f"|regret@{k}={metrics[f'regret@{k}']:.4f}"
        f"|hit@{k}={metrics[f'hit@{k}'] * 100:.2f}%"
        # f" | within1dB@{k}={metrics[f'within_1dB@{k}'] * 100:.2f}%"

    return msg


def compute_metrics_from_regrets_and_hits(regrets, hits, k_list):
    metrics = {}

    for k in k_list:
        r = np.asarray(regrets[k], dtype=np.float64)
        h = np.asarray(hits[k], dtype=np.float64)

        metrics[f"regret@{k}"] = float(np.mean(r))
        # metrics[f"regret_median@{k}"] = float(np.median(r))
        # metrics[f"regret_p75@{k}"] = float(np.quantile(r, 0.75))
        # metrics[f"regret_p90@{k}"] = float(np.quantile(r, 0.90))
        # metrics[f"regret_p95@{k}"] = float(np.quantile(r, 0.95))
        # metrics[f"regret_max@{k}"] = float(np.max(r))
        #
        metrics[f"hit@{k}"] = float(np.mean(h))
        # metrics[f"within_0.5dB@{k}"] = float(np.mean(r <= 0.5))
        # metrics[f"within_1dB@{k}"] = float(np.mean(r <= 1.0))
        # metrics[f"within_2dB@{k}"] = float(np.mean(r <= 2.0))

    return metrics


def evaluate_score_array(score, power, k_list):
    """
    Numpy-only top-K evaluation, useful for non-torch baselines.

    score: [N, 4941], larger is better
    power: [N, 4941], true received power in dB
    """
    max_k = max(k_list)

    regrets = {k: [] for k in k_list}
    hits = {k: [] for k in k_list}

    true_best_idx = np.argmax(power, axis=1)
    true_best_power = power[np.arange(power.shape[0]), true_best_idx]

    topk_unsorted = np.argpartition(score, -max_k, axis=1)[:, -max_k:]
    topk_scores = np.take_along_axis(score, topk_unsorted, axis=1)
    order = np.argsort(topk_scores, axis=1)[:, ::-1]
    pred_topk = np.take_along_axis(topk_unsorted, order, axis=1)

    for i in range(power.shape[0]):
        for k in k_list:
            idx_k = pred_topk[i, :k]
            selected_power = power[i, idx_k].max()
            regret = true_best_power[i] - selected_power
            hit = float(true_best_idx[i] in idx_k)

            regrets[k].append(float(regret))
            hits[k].append(hit)

    return compute_metrics_from_regrets_and_hits(regrets, hits, k_list)


def print_compact_final_metrics(prefix, metrics):
    print(
        f"{prefix} | "
        f"regret@1={metrics['regret@1']:.4f} | "
        f"regret@5={metrics['regret@5']:.4f} | "
        f"regret@10={metrics['regret@10']:.4f} | "
        f"regret@15={metrics['regret@15']:.4f} | "
        f"regret@20={metrics['regret@20']:.4f} | "
        f"hit@1={metrics['hit@1']:.4f} | "
        f"hit@5={metrics['hit@5']:.4f} | "
        f"hit@10={metrics['hit@10']:.4f} | "
        f"hit@15={metrics['hit@15']:.4f} | "
        f"hit@20={metrics['hit@20']:.4f} | "
    )