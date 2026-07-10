import torch
import numpy as np

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def power_to_gap_torch(power):
    best_power = power.max(dim=1, keepdim=True).values
    gap = power - best_power
    return gap




def average_metrics_over_checkpoints(final_test_by_selected_k):

    avg_test_metrics = {}
    metric_keys = list(next(iter(final_test_by_selected_k.values())).keys())

    for metric_key in metric_keys:
        values = [
            metrics[metric_key]
            for metrics in final_test_by_selected_k.values()
            if metric_key in metrics
        ]
        avg_test_metrics[metric_key] = float(np.mean(values))

    return avg_test_metrics


def best_regrets_from_testdataset(final_test_by_selected_k, k_list):
    best_metrics = {}
    best_source = {}

    for k in k_list:
        key = f"regret@{k}"

        candidates = [
            (monitor_k, test_metrics[key])
            for monitor_k, test_metrics in final_test_by_selected_k.items()
            if key in test_metrics
        ]

        best_monitor_k, best_value = min(candidates, key=lambda item: item[1])

        best_metrics[key] = float(best_value)
        best_source[key] = best_monitor_k

    return best_metrics, best_source


def best_accuracy_from_testdataset(final_test_by_selected_k, k_list):
    best_metrics = {}
    best_source = {}

    for k in k_list:
        key = f"hit@{k}"

        candidates = [
            (monitor_k, test_metrics[key])
            for monitor_k, test_metrics in final_test_by_selected_k.items()
            if key in test_metrics
        ]

        best_monitor_k, best_value = max(candidates, key=lambda item: item[1])

        best_metrics[key] = float(best_value)
        best_source[key] = best_monitor_k

    return best_metrics, best_source