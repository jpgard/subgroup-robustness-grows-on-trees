"""
Utility functions for model training.
"""

import logging
from multiprocessing import Pool
from typing import Sequence

import wandb

import src.torchutils
from src.datasets import Dataset
import src.experiment_utils
from src.config import CONFIG_FNS
from src.utils import BackEndConfig


def train_model(config_file, dset: Dataset, device):
    run = wandb.init(project="disparity-experiments",
                     reinit=True, config=config_file)
    with run:
        logging.info(f"fitting model from {config_file}")
        config = wandb.config
        model_type = config.model_type
        config_parse_fn = CONFIG_FNS[model_type]
        criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs = \
            config_parse_fn(config)

        fit_model = src.experiment_utils.get_and_fit_model(
            dset,
            kind=model_type,
            is_regression=dset.is_regression,
            criterion_kwargs=criterion_kwargs,
            model_kwargs=model_kwargs,
            optimizer=config.get("optimizer", None),
            opt_kwargs=opt_kwargs,
            device=device,
            **fit_kwargs)

        model_uid = f"{config.model_type}-{config.model_tag}"
    return model_uid, fit_model


def train_models(dset: Dataset, model_configs: Sequence[str],
                 parallel_config: BackEndConfig):
    """
    Train a set of models, by name, on src_dataset.

    :return: a dictionary mapping a set of string model UIDs to trained models,
        one entry for each config file in model_configs.
    """
    device = src.torchutils.get_device(parallel_config.disable_cuda)
    trained_models = dict()
    if parallel_config.no_parallel:
        for config_file in model_configs:
            model_uid, fit_model = train_model(config_file, dset=dset,
                                               device=device)
            trained_models[model_uid] = fit_model
    else:
        async_res = []
        with Pool(parallel_config.n_jobs) as pool:
            for config_file in model_configs:
                async_res.append(pool.apply_async(
                    train_model,
                    kwds={"config_file": config_file,
                          "dset": dset,
                          "device": device}))
            results = [r.get() for r in async_res]
        trained_models = dict(results)

    return trained_models
