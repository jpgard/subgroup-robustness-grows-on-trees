"""
Tools to compute descriptive statistics for a dataset.
"""

import logging
import pandas as pd
from src.utils import LOG_LEVEL


logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')

def compute_descriptive_metrics(X: pd.DataFrame, y: pd.DataFrame):
    metrics = {}
    metrics["n"] = X.shape[0]
    metrics["d"] = X.shape[1]
    try:
        metrics["majority_fraction"] = X['sensitive'].mean()
    except Exception as e:
        logger.warning(e)
    return metrics
