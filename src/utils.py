"""
Various utility functions.
"""

import hashlib
from datetime import datetime
from dataclasses import dataclass
import logging
import os

import numpy as np
import pandas as pd
import torch

LOG_LEVEL = logging.INFO


@dataclass
class BackEndConfig:
    """Class for keeping track of back-end configuration."""
    no_parallel: bool
    n_jobs: int
    disable_cuda: bool
    gpu_id: int


def get_results_filepath(dirname=".", suffix="", **kwargs) -> str:
    start_datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    fname_suffix = ""
    for k, v in kwargs.items():
        fname_suffix += "_".join((k, v))
    if suffix != "":
        suffix = "_" + suffix
    filename = f"results_{start_datetime}{fname_suffix}{suffix}.csv"
    return os.path.join(dirname, filename)


def subset_by_value(X: pd.DataFrame, y: pd.Series,
                    subset_value, subset_colname='sensitive',
                    return_idxs=False):
    idxs = X[subset_colname] == subset_value
    if not return_idxs:
        return X.loc[idxs], y.loc[idxs]
    else:
        return X.loc[idxs], y.loc[idxs], idxs


def divide_nan_safe(a, b):
    if not np.isnan(b):
        return a / b
    else:
        return np.nan


def assert_no_nan(inputs):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.numpy()
    assert not np.any(np.isnan(inputs))
    return


def make_uid():
    return hashlib.md5(str(datetime.now).encode()).hexdigest()
