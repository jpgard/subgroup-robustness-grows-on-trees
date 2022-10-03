"""
Utilities for parsing .yaml configs.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import yaml
from sklearn.kernel_approximation import Nystroem, RBFSampler
from src import *
from src.utils import LOG_LEVEL

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')

# Specific configs dropped due to extremely slow runtimes
# + no performance benefit.

EXCLUDED_CONFIGS = [
    {"geometry": "chi-square",
     "size": 1.0,
     "model_type": "fastdro"},
    {"geometry": "chi-square",
     "learning_rate": 0.00001,
     "model_type": "fastdro"},
    {"geometry": "chi-square",
     "learning_rate": 0.0001,
     "model_type": "fastdro"},
    {"geometry": "chi-square",
     "learning_rate": 0.001,
     "model_type": "fastdro"},
]


def is_excluded_config(config) -> bool:
    """Check whether a config is excluded."""
    for to_exclude in EXCLUDED_CONFIGS:
        if all(key in config and config[key] == to_exclude[key] for key in
               to_exclude):
            logging.warning("dropping excluded config: {}".format(config))
            return True
    return False


def grid_size(params):
    """Find the total size of a grid search.

    Expects a structure matching the output of read_yaml().
    """
    return np.prod([len(param_grid["values"]) for param_grid in
                    params["parameters"].values()])


def read_yaml(yaml_file):
    """Read a .yaml file."""
    with open(yaml_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params


def parse_opt_kwargs(config: dict) -> dict:
    """Parse a set of kwargs shared across all torch models for optimization."""
    assert "optimizer" in config
    opt = config["optimizer"]
    if opt == "sgd":
        return {"lr": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "momentum": config["momentum"]}
    elif opt == "adamw":
        return {"lr": config["learning_rate"],
                "weight_decay": config["weight_decay"]}
    elif opt == "qhadam":
        return {"lr": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                'nus': config['nus'],
                'betas': config['betas']}
    else:
        raise NotImplementedError("optimizer %s not implemented" % opt)


def parse_mlp_fit_kwargs(config: dict) -> dict:
    """Parse a set of kwargs shared across MLP-based models for fitting."""
    return {"steps": config.get("steps"),
            "epochs": config.get("epochs"),
            "batch_size": config["batch_size"]}


def parse_mlp_model_kwargs(config: dict) -> dict:
    """Parse a set of architectural kwargs shared across MLP-based models."""
    return {"num_layers": config["num_layers"],
            "d_hidden": config["d_hidden"],
            "dropout_prob": config.get("dropout_prob")}


def unpack_config_mwld(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"],
                        'lv_lambda': config['lv_lambda'], }
    opt_kwargs = {"lr": config["learning_rate"],
                  "weight_decay": config["l2_eta"],
                  "momentum": config["momentum"]}
    fit_kwargs = parse_mlp_fit_kwargs(config)
    model_kwargs = parse_mlp_model_kwargs(config)
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_marginal_dro(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {
        x: config[x] for x in ("criterion_name", "radius")}
    opt_kwargs = parse_opt_kwargs(config)
    fit_kwargs = parse_mlp_fit_kwargs(config)
    model_kwargs = parse_mlp_model_kwargs(config)
    model_kwargs["p_min"] = config["p_min"]
    model_kwargs["niter_inner"] = config["niter_inner"]
    model_kwargs["nbisect"] = config["nbisect"]
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_fastdro(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"],
                        "geometry": config["geometry"],
                        "size": config["size"],
                        "reg": config["reg"],
                        "max_iter": config["max_iter"]}
    opt_kwargs = parse_opt_kwargs(config)
    fit_kwargs = parse_mlp_fit_kwargs(config)
    model_kwargs = parse_mlp_model_kwargs(config)
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_randomforest(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {
        "n_estimators": config["n_estimators"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": config["max_features"],
        "ccp_alpha": config["ccp_alpha"],
    }
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_doro(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"],
                        "geometry": config["geometry"],
                        "alpha": config["alpha"],
                        "eps": config["eps"]}
    opt_kwargs = parse_opt_kwargs(config)
    fit_kwargs = parse_mlp_fit_kwargs(config)
    model_kwargs = parse_mlp_model_kwargs(config)
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_group_dro(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {"criterion_name": config["criterion_name"],
                        "group_weights_step_size": config[
                            "group_weights_step_size"]}
    opt_kwargs = parse_opt_kwargs(config)
    fit_kwargs = parse_mlp_fit_kwargs(config)
    model_kwargs = parse_mlp_model_kwargs(config)
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_mlp(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {'criterion_name': config['criterion_name']}
    opt_kwargs = parse_opt_kwargs(config)
    fit_kwargs = parse_mlp_fit_kwargs(config)
    model_kwargs = parse_mlp_model_kwargs(config)
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_gbm(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {
        'learning_rate': config['learning_rate'],
        'n_estimators': config['n_estimators'],
        'min_samples_split': config['min_samples_split'],
        'min_samples_leaf': config['min_samples_leaf'],
        'max_depth': config['max_depth'],
        'max_features': config['max_features'],
    }
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_histgbm(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {
        'learning_rate': config['learning_rate'],
        'max_bins': config['max_bins'],
        'max_leaf_nodes': config['max_leaf_nodes'],
        'min_samples_leaf': config['min_samples_leaf'],
        'l2_regularization': config['l2_regularization'],
    }
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def fetch_gbm_configs(config) -> dict:
    model_kwargs = {}
    for k in ('learning_rate', 'n_estimators', 'n_estimators',
              'min_samples_split', 'min_samples_leaf', 'min_samples_leaf',
              'max_depth', 'max_features'):
        if k in config:
            model_kwargs[k] = config[k]
    return model_kwargs


def unpack_config_expgrad(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {'constraint': config['constraint']}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {'base_learner': config['base_learner'],
                    'base_learner_kwargs': fetch_gbm_configs(config),
                    "eps": config["eps"],
                    "eta0": config["eta0"],
                    "max_iter": config["max_iter"]
                    }
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_lfr_preprocess(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {'Ax': config['Ax'],
                        'Ay': config['Ay'],
                        'Az': config['Az'],
                        'k': config['k'], }
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {'base_learner': config['base_learner'],
                    'maxiter': config['maxiter'],
                    'maxfun': config['maxfun'],
                    'base_learner_kwargs': fetch_gbm_configs(config)}

    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_eo_postprocess(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {'base_learner': config['base_learner'],
                    'base_learner_kwargs': fetch_gbm_configs(config),
                    'postprocessor_constraint': config[
                        'postprocessor_constraint']}
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_svm(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    kernel_kwargs = {
        "gamma": config["gamma"],
        "n_components": config["n_components"],
    }
    if config["kernel_type"] == "nystroem":
        kernel_kwargs["degree"] = config["nystroem_kernel_degree"]
        kernel_kwargs["kernel"] = config["nystroem_kernel"]

    model_kwargs = {"C": config["C"],
                    "kernel_kwargs": kernel_kwargs,
                    "loss": config["loss"]
                    }
    if config["kernel_type"] == "nystroem":
        model_kwargs["kernel_fn"] = Nystroem
    elif config["kernel_type"] == "rks":
        model_kwargs["kernel_fn"] = RBFSampler
    else:
        raise NotImplementedError
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_l2lr(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {"C": config["C"]}
    if "class_weight" in config:
        model_kwargs.update({"class_weight": config["class_weight"]})
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_lightgbm(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "n_estimators": config["n_estimators"],
        "reg_lambda": config["reg_lambda"],
        "min_child_samples": config["min_child_samples"],
    }
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_xgb(config) -> Tuple[dict, dict, dict, dict]:
    criterion_kwargs = {}
    opt_kwargs = {}
    fit_kwargs = {}
    model_kwargs = {
        "learning_rate": config["learning_rate"],
        "min_split_loss": config["min_split_loss"],
        "max_depth": config["max_depth"],
        "colsample_bytree": config["colsample_bytree"],
        "colsample_bylevel": config["colsample_bylevel"],
        "max_bin": config["max_bin"],
        "grow_policy": config["grow_policy"]
    }
    return criterion_kwargs, opt_kwargs, fit_kwargs, model_kwargs


def unpack_config_lr(config):
    if "class_weight" in config:
        model_kwargs = {"class_weight": config["class_weight"]}
    else:
        model_kwargs = dict()
    return dict(), dict(), dict(), model_kwargs


def null_config(unused_config) -> Tuple[dict, dict, dict, dict]:
    return dict(), dict(), dict(), dict()


# Dictionary mapping model type (string) to a Callable that parses configs
# for the model. The Callable returns a tuple of four dictionaries,
# containing kwargs for the criterion, optimizer, fit, and model respectively.

CONFIG_FNS = {
    DORO_MODEL: unpack_config_doro,
    POSTPROCESSOR_MODEL: unpack_config_eo_postprocess,
    EXPGRAD_MODEL: unpack_config_expgrad,
    FAST_DRO_MODEL: unpack_config_fastdro,
    GBM_MODEL: unpack_config_gbm,
    GROUP_DRO_MODEL: unpack_config_group_dro,
    HISTGBM_MODEL: unpack_config_histgbm,
    IWC_MODEL: null_config,
    LFR_PREPROCESSOR_MODEL: unpack_config_lfr_preprocess,
    LIGHTGBM_MODEL: unpack_config_lightgbm,
    L2_MODEL: unpack_config_l2lr,
    LR_MODEL: unpack_config_lr,
    MARGINAL_DRO_MODEL: unpack_config_marginal_dro,
    MLP_MODEL: unpack_config_mlp,
    MWLD_MODEL: unpack_config_mwld,
    RANDOM_FOREST_MODEL: unpack_config_randomforest,
    SVM_MODEL: unpack_config_svm,
    XGBOOST_MODEL: unpack_config_xgb,
}
