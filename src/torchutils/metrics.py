"""
Utility functions to compute metrics for torch models.
"""

import logging
import math

import pandas as pd
import torch
from torch.nn.functional import mse_loss

from src.torchutils import pd_to_torch_float
from src.utils import LOG_LEVEL

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def subgroup_metric(preds, labels, g, group_label: int,
                    metric_fn) -> torch.Tensor:
    """Compute the mean value of metric_fn over instances where g==group_label.
    """
    per_element_metric = metric_fn(preds, labels)
    g_mask = (g == group_label).double()
    assert len(per_element_metric) == len(g_mask), "shape sanity check"
    return torch.sum(per_element_metric * g_mask) / torch.sum(g_mask)


def compute_regression_disparity_metrics(preds, labels, sens, prefix=""):
    if prefix != "":
        prefix += "_"
    if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
        labels = pd_to_torch_float(labels)
    metrics = {}
    for g in (0, 1):
        mse_g = subgroup_metric(preds, labels, sens, group_label=g,
                                metric_fn=mse_loss)
        metrics[f"mse_{g}"] = mse_g
        metrics[f"rmse_{g}"] = math.sqrt(mse_g)
    for metric in ("mse", "rmse"):
        metrics[f"{metric}_disparity"] = \
            metrics[f"{metric}_1"] - metrics[f"{metric}_0"]
        metrics[f"{metric}_abs_disparity"] = \
            abs(metrics[f"{metric}_1"] - metrics[f"{metric}_0"])
        metrics[f"{metric}_worstgroup"] = \
            max(metrics[f"{metric}_1"], metrics[f"{metric}_0"])
    return metrics


def binary_accuracy_from_probs(inputs: torch.Tensor, labels: torch.Tensor,
                               reduction="mean"):
    assert torch.all(inputs >= 0.) and torch.all(
        inputs <= 1.), "expected probs in range[0.,1.]"
    hard_preds = (inputs > 0.5).float()
    binary_correctness = (labels == hard_preds).float()
    if reduction == "mean":
        return torch.mean(binary_correctness)
    elif reduction == "none":
        return binary_correctness
    else:
        raise NotImplementedError(f"reduction {reduction} not implemented")
