"""
Utilities for working with aif360 models.
"""

import logging
from typing import Tuple, List, Optional

from aif360.datasets.structured_dataset import StructuredDataset
from aif360.datasets.binary_label_dataset import BinaryLabelDataset
import pandas as pd

from src.datasets import TGT
from src.utils import LOG_LEVEL

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def structured_dataset_to_pandas(
        dataset: StructuredDataset,
        feature_names: Optional[List[str]] = None,
        label_name: str = TGT) -> Tuple[
    pd.DataFrame, pd.Series]:
    """Convert a structured dataset to a (features, labels) tuple."""
    X = pd.DataFrame(dataset.features, columns=feature_names)
    y = pd.Series(dataset.labels.ravel(), name=label_name)
    return X, y


def structured_dataset_to_dataframe(
        dataset: StructuredDataset,
        feature_names: Optional[List[str]] = None,
        label_name: str = TGT) -> pd.DataFrame:
    """Convert a structured dataset to a DataFrame."""
    X, y = structured_dataset_to_pandas(dataset=dataset,
                                        feature_names=feature_names,
                                        label_name=label_name)
    return pd.concat((X, y), axis=1)


def to_structured_dataset(df: pd.DataFrame,
                          protected_attribute_names: Tuple[str],
                          label_name=TGT,
                          ) -> StructuredDataset:
    """Convert a DataFrame to a StructuredDataset."""
    tmp = StructuredDataset(
        df, label_names=(label_name,),
        protected_attribute_names=protected_attribute_names)
    return tmp


def to_binary_label_dataset(
        df: pd.DataFrame,
        protected_attribute_names: Tuple[str],
        label_name=TGT,
        favorable_label=1., unfavorable_label=0.) -> BinaryLabelDataset:
    """Convert a DataFrame to a BinaryLabelDataset."""
    return BinaryLabelDataset(
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label,
        df=df, label_names=(label_name,),
        protected_attribute_names=protected_attribute_names)
