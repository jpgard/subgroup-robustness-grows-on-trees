from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader

# SENS = "sensitive"  # name of sensitive column
TGT = "target"  # name of target column
DATASET_ROOT = "./datasets"
DEFAULT_TRAIN_FRAC = 0.8  # remaining data split evenly between val/test
SPLIT_RANDOM_STATE = 75534  # optional fixed random state for train-test splits

SENS_RACE = "race"
SENS_SEX = "sex"
SENS_AGE = "age"


@dataclass
class DatasetConfig:
    sens: list
    batch_size: int = None
    root_dir: str = DATASET_ROOT
    use_cache: bool = True
    random_state: int = SPLIT_RANDOM_STATE
    # float in range (0,1]; fraction of the full dataset to use (over all splits).
    subsample_size: float = 1.
    # float in range (0,1)
    train_frac: float = DEFAULT_TRAIN_FRAC


class Dataset(ABC):
    @abstractmethod
    def get_dataset_root_dir(self):
        raise

    @abstractmethod
    def get_dataloader(self, split="train", shuffle=True, drop_last=False,
                       batch_size: int = None) -> DataLoader:
        raise

    @property
    @abstractmethod
    def d(self):
        raise

    def get_privileged_and_unprivileged_groups(self) -> Tuple[
        Tuple[dict], Tuple[dict]]:
        """Fetch privileged and unprivileged groups for aif360-based models."""
        privileged_groups = ({x: 1 for x in self.sens},)
        unprivileged_groups = ({x: 0 for x in self.sens},)
        return privileged_groups, unprivileged_groups

    @abstractmethod
    def _load(self):
        raise

    @abstractmethod
    def get_data(self):
        raise

    @abstractmethod
    def get_data_with_groups(self):
        raise
