from dataclasses import asdict

from src.datasets import tabular, DatasetConfig
from src.datasets.larc_utils import GRADE_TASK, RETENTION_TASK


def get_dataset(config: DatasetConfig):
    if isinstance(config, tabular.ACSDatasetConfig):
        return tabular.ACSDataset(**asdict(config))
    elif isinstance(config, tabular.AdultDatasetConfig):
        return tabular.AdultDataset(**asdict(config))
    elif isinstance(config, tabular.CompasDatasetConfig):
        return tabular.CompasDataset(**asdict(config))
    elif isinstance(config, tabular.GermanDatasetConfig):
        return tabular.GermanDataset(**asdict(config))
    elif isinstance(config, tabular.CommunitiesAndCrimeDatasetConfig):
        return tabular.CommunitiesAndCrimeDataset(**asdict(config))
    elif isinstance(config, tabular.BRFSSDatasetConfig):
        return tabular.BRFSSDataset(**asdict(config))
    elif isinstance(config, tabular.LARCDatasetConfig):
        if config.larc_task == GRADE_TASK:
            return tabular.LARCAtRiskDataset(**asdict(config))
    elif isinstance(config, tabular.MOOCDatasetConfig):
        return tabular.MOOCDataset(**asdict(config))
    raise NotImplementedError
