"""
Train a model with a fixed set of hyperparameters.

This is also the script called by the wandb agent during a sweep.
"""
import argparse
import json
import logging
import time

import wandb

import src.datasets.tabular
import src.datasets.utils
import src.torchutils
from src import experiment_utils
from src.config import CONFIG_FNS, is_excluded_config
from src.datasets.tabular import get_dataset_config
from src.utils import LOG_LEVEL
from src import evaluation

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def main(dataset: str,
         dataset_kwargs: str,
         default_config: str = "./configs/mwld.yaml",
         disable_cuda: bool = False,
         print_eval_results: bool = False):
    start = time.time()
    device = src.torchutils.get_device(disable_cuda)
    dataset_kwargs = json.loads(dataset_kwargs)

    wandb.init(config=default_config)
    config = wandb.config

    if is_excluded_config(config):
        # This is a config to drop from the sweep; skip it.
        return

    logging.info(f"dataset name is {dataset}")
    dataset_config = get_dataset_config(dataset, **dataset_kwargs)
    dset = src.datasets.utils.get_dataset(dataset_config)
    config.dataset = dset.name
    config.make_dummies = dataset_config.make_dummies
    config.label_encode_categorical_cols = dataset_config.label_encode_categorical_cols

    # unpack config into criterion and fit kwargs
    model_type = config["model_type"]

    criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs = CONFIG_FNS[
        model_type](config)

    model = experiment_utils.get_model(
        dset=dset,
        kind=model_type,
        model_kwargs=model_kwargs,
        device=device,
        criterion_kwargs=criterion_kwargs)
    if "optimizer" in config:
        opt = src.torchutils.get_optimizer(config["optimizer"],
                                           model, **opt_kwargs)
        fit_kwargs.update({"optimizer": opt})
    fit_model = experiment_utils.fit_model(
        dset,
        model=model,
        is_regression=dataset_config.is_regression,
        **fit_kwargs)
    if print_eval_results:
        _, _, X_te, y_te, _, _ = dset.get_data()
        metrics = evaluation.fetch_eval_metrics(
            fit_model, X_te, y_te, is_regression=dataset_config.is_regression)
        print(metrics)

    logging.info("finished in {}s".format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='income')
    parser.add_argument(
        '--dataset_kwargs',
        default='{"root_dir": "datasets", "make_dummies": true}',
        type=str,
        help='json-formatted string of arguments to the'
             'dataset class constructor.')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument(
        '--default_config',
        help="path to a .yaml file to use for training",
        default=None)
    parser.add_argument('--print_eval_results', default=False,
                        action='store_true')
    args, unknown = parser.parse_known_args()
    main(**vars(args))
