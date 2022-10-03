"""
Utilities for evaluating model performance.
"""

import logging

import pandas as pd
import wandb

from src.torchutils.models import DenseModel
from src.metrics import metrics_by_group
from src.utils import LOG_LEVEL, assert_no_nan
from src.models import ModelWithPreprocessor

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def fetch_eval_metrics(model, X_te: pd.DataFrame = None,
                       y_te: pd.DataFrame = None,
                       sensitive_features: pd.DataFrame = None,
                       loader=None,
                       is_regression: bool = False, suffix='') -> dict:
    if suffix != '':
        suffix = "_" + suffix

    if (X_te is None) and isinstance(model, DenseModel):
        assert loader is not None
        yhat_soft, y_te, sens = model.predict_on_loader(loader)
        yhat_soft = yhat_soft.detach().cpu().numpy()
        y_te = y_te.detach().cpu().numpy()
        sens = sens.detach().cpu().numpy()
        yhat_hard = (yhat_soft > 0.5).astype(int)

    else:
        sens = X_te[sensitive_features]
        # sklearn-style model; fetch predictions
        if isinstance(model, ModelWithPreprocessor):
            X_te = model.preprocess(X_te, y_te)
        yhat_hard = model.predict(X_te)
        yhat_soft = model.predict_proba(X_te)

    if isinstance(y_te, pd.Series):
        y_te = y_te.values

    try:
        assert_no_nan(yhat_soft)
        assert_no_nan(yhat_hard)
    except AssertionError:
        logging.error("invalid predictions; skipping classification"
                      f" metrics for {model}")
        return {}

    metrics = metrics_by_group(y_true=y_te, yhat_soft=yhat_soft,
                               yhat_hard=yhat_hard,
                               sensitive_features=sens, suffix=suffix)
    return metrics


def log_eval_metrics(model, X, y,
                     sensitive_features=None,
                     is_regression=False,
                     loader=None,
                     suffix='val'):
    """Log metrics for the test set."""
    assert (X is not None and y is not None) or (loader is not None)
    metrics = fetch_eval_metrics(model, X_te=X, y_te=y,
                                 sensitive_features=sensitive_features,
                                 is_regression=is_regression,
                                 loader=loader,
                                 suffix=suffix)
    wandb.log(metrics)
    return
