"""
Tools to compute various train/eval metrics.
"""
from collections.abc import Iterable
import functools
import logging

import fairlearn.metrics
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as skm
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
import torch
from torch.nn.functional import binary_cross_entropy
from scipy import interpolate
from scipy import integrate

from src.torchutils import safe_cast_to_numpy, clip_torch_outputs
from src.torchutils.criterion import cvar_doro_criterion
from src.utils import LOG_LEVEL

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def extract_positive_class_predictions(y_pred):
    if len(y_pred.shape) == 2:
        # Case: predictions contain class preds for classes (0,1).
        return y_pred[:, 1]
    else:
        # Case: predictions contain class preds only for class 1.
        return y_pred


def cvar_doro_metric(y_true, y_pred, eps=0.005, alpha=0.2) -> float:
    """Compute CVaR DORO metric with a fairlearn-compatible interface."""
    y_pred = extract_positive_class_predictions(y_pred)
    outputs_clipped = clip_torch_outputs(torch.from_numpy(y_pred).double())
    targets_clipped = clip_torch_outputs(torch.from_numpy(y_true).double())

    return cvar_doro_criterion(outputs=outputs_clipped,
                               targets=targets_clipped,
                               eps=eps,
                               alpha=alpha).detach().cpu().numpy().item()


def cvar_metric(y_true, y_pred, alpha=0.2) -> float:
    """Compute CVaR metric with a fairlearn-compatible interface."""
    y_pred = extract_positive_class_predictions(y_pred)
    outputs_clipped = clip_torch_outputs(torch.from_numpy(y_pred).double())
    targets_clipped = clip_torch_outputs(torch.from_numpy(y_true).double())
    return cvar_doro_criterion(outputs=outputs_clipped,
                               targets=targets_clipped,
                               eps=0.,
                               alpha=alpha).detach().cpu().numpy().item()


def loss_variance_metric(y_true, y_pred):
    """Compute loss variance metric with a fairlearn-compatible interface."""
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, 1]
    elementwise_loss = binary_cross_entropy(
        input=torch.from_numpy(y_pred).float(),
        target=torch.from_numpy(y_true).float(),
        reduction="none")
    return torch.var(elementwise_loss).cpu().numpy().item()


def all_subgroups_contain_all_label_values(y_true, sens) -> bool:
    """Check whether all labels are represented in all sensitive subgroups."""
    if np.ndim(sens) == 2:
        # Case: multiple sensitive attributes
        sens_cols = [sens.iloc[:, i] for i in range(sens.shape[1])]
        crosstab = pd.crosstab(y_true, sens_cols)
    else:
        # Case: single sensitive attribute
        crosstab = pd.crosstab(y_true, sens)
    return np.any(crosstab == 0)


def append_suffix_to_keys(d: dict, suffix: str) -> dict:
    return {f"{k}{suffix}": v for k, v in d.items()}


def _log_intersectional_metrics(grouped_metrics, metrics, suffix,
                                sensitive_features):
    # Compute metrics by subgroup (intersectional); note that marginals can
    # always be recovered using the per-group counts.
    for sens_idx, metrics_dict in grouped_metrics.by_group.to_dict(
            'index').items():
        if not isinstance(sens_idx, Iterable):
            # Case: only a single sensitive attribute.
            sens_idx = [sens_idx]
        # sens_Str is e.g. 'race0sex1'
        sens_str = ''.join(f"{col}{val}" for col, val in
                           zip(sensitive_features.columns, sens_idx))

        for metric_name, metric_value in metrics_dict.items():
            metrics[sens_str + metric_name + suffix] = metric_value
    return metrics


def metrics_by_group(y_true, yhat_soft, yhat_hard,
                     sensitive_features: pd.DataFrame, suffix: str = ''):
    assert isinstance(sensitive_features, pd.DataFrame)
    if (suffix != '') and (not suffix.startswith('_')):
        # Ensure suffix has proper leading sep token
        suffix = '_' + suffix
    metrics = {}
    _log_loss = functools.partial(skm.log_loss, labels=[0., 1.])

    metric_fns_continuous = {
        'crossentropy': _log_loss,
        'cvar_doro': cvar_doro_metric,
        'cvar': cvar_metric,
        'loss_variance': loss_variance_metric,
    }

    metric_fns_binary = {
        'accuracy': skm.accuracy_score,
        'selection_rate': fairlearn.metrics.selection_rate,
        'count': fairlearn.metrics.count,
        'tpr': fairlearn.metrics.true_positive_rate,
        'fpr': fairlearn.metrics.false_positive_rate,
    }

    if all_subgroups_contain_all_label_values(y_true, sensitive_features):
        # Only compute AUC if all labels exist in each sens group. This is
        # due to a limitation in fairlearn.MetricFrame, which can't handle
        # errors or nan values when computing group difference metrics.
        metric_fns_binary['auc'] = skm.roc_auc_score
    else:
        logging.info("Not computing AUC for this split because one or more"
                     " sensitive subgroups do not contain all classes.")

    grouped_metrics_binary = fairlearn.metrics.MetricFrame(
        metrics=metric_fns_binary,
        y_true=y_true,
        y_pred=yhat_hard,
        sensitive_features=sensitive_features)

    grouped_metrics_continuous = fairlearn.metrics.MetricFrame(
        metrics=metric_fns_continuous,
        y_true=y_true,
        y_pred=yhat_soft,
        sensitive_features=sensitive_features)

    metrics.update(append_suffix_to_keys(
        grouped_metrics_binary.overall.to_dict(), suffix))

    metrics.update(append_suffix_to_keys(
        grouped_metrics_continuous.overall.to_dict(), suffix))

    for metric_name, metric_value in grouped_metrics_continuous.difference().iteritems():
        metrics[f"abs_{metric_name}_disparity{suffix}"] = metric_value

    # Compute some specific metrics of interest from the results
    metrics['abs_accuracy_disparity' + suffix] = \
        grouped_metrics_binary.difference()['accuracy']
    metrics['demographic_parity_diff' + suffix] = \
        grouped_metrics_binary.difference()['selection_rate']
    # EO diff is defined as  The greater of two metrics:
    # `true_positive_rate_difference` and `false_positive_rate_difference`;
    # see fairlearn.metrics.equalized_odds_difference
    metrics['equalized_odds_diff' + suffix] = \
        max(grouped_metrics_binary.difference()[['tpr', 'fpr']])

    metrics["accuracy_worstgroup" + suffix] = \
        grouped_metrics_binary.group_min()['accuracy']
    metrics["crossentropy_worstgroup" + suffix] = \
        grouped_metrics_continuous.group_max()['crossentropy']
    metrics['cvar_doro_worstgroup' + suffix] = \
        grouped_metrics_continuous.group_max()['cvar_doro']
    metrics['cvar_worstgroup' + suffix] = \
        grouped_metrics_continuous.group_max()['cvar']

    metrics = _log_intersectional_metrics(grouped_metrics_binary, metrics,
                                          suffix, sensitive_features)
    metrics = _log_intersectional_metrics(grouped_metrics_continuous, metrics,
                                          suffix, sensitive_features)

    return metrics
