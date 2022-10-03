"""
Utilities for running experiments.
"""

import logging
from typing import Optional

import numpy as np
from aif360.algorithms.preprocessing import LFR
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from fairlearn.reductions import DemographicParity, EqualizedOdds, \
    ErrorRateParity
import sklearn

# HistGradientBoosting classes are experimental for sklearn < 1.x, which are
# the only versions compatible with Python 3.6.
import src.torchutils
import src.torchutils.criterion

if not sklearn.__version__.startswith('1'):
    from sklearn.experimental import enable_hist_gradient_boosting

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, \
    GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn import linear_model

from src import *
from src.datasets import Dataset
from src.datasets.tabular import TabularDataset
from src.models import CustomExponentiatedGradient, ModelWithPostprocessor, \
    ModelWithLFRPreprocessor, SVMWithKernelApprox, \
    WeightedCovariateShiftClassifier, CustomThresholdOptimizer
from src.torchutils.models import DenseModel, GroupDROModel, \
    DenseModelWithLossParams
from src.evaluation import log_eval_metrics

# Regularization parameter grid for cross-validated L2 logistic regression.
L2LAMBDA_GRID = np.logspace(-10, 10, num=21)


def get_classifier(dset: Dataset, kind, device=None, criterion_kwargs=None,
                   model_kwargs=None, ):
    """Instantiate and return a classifier with the specified parameters."""
    categorical_columns_meta = getattr(dset, "categorical_columns_meta",
                                       None)
    d = dset.d
    (privileged_groups,
     unprivileged_groups) = dset.get_privileged_and_unprivileged_groups()

    if criterion_kwargs is None:
        criterion_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}

    elif kind == DORO_MODEL:
        assert criterion_kwargs[
                   "criterion_name"] == src.torchutils.criterion.DORO_CRITERION
        model = DenseModel(dset.d, device,
                           is_regressor=False,
                           criterion_kwargs=criterion_kwargs,
                           model_type=DORO_MODEL,
                           **model_kwargs)

    elif kind == EXPGRAD_MODEL:
        # Base learner
        base_learner_kind = model_kwargs.pop("base_learner")
        base_learner_kwargs = model_kwargs.pop("base_learner_kwargs")
        base_learner = get_classifier(dset, base_learner_kind,
                                      model_kwargs=base_learner_kwargs)
        # Constraint
        if criterion_kwargs["constraint"] == DP_CONSTRAINT:
            constraint = DemographicParity()
        elif criterion_kwargs["constraint"] == EO_CONSTRAINT:
            constraint = EqualizedOdds()
        elif criterion_kwargs["constraint"] == ERROR_RATE_PARITY_CONSTRAINT:
            constraint = ErrorRateParity()
        else:
            raise NotImplementedError
        # Model
        model = CustomExponentiatedGradient(estimator=base_learner,
                                            constraints=constraint,
                                            sensitive_features=dset.sens,
                                            **model_kwargs)

    elif kind == FAST_DRO_MODEL:
        assert criterion_kwargs[
                   "criterion_name"] == src.torchutils.criterion.FASTDRO_CRITERION
        model = DenseModel(d, device,
                           is_regressor=False,
                           criterion_kwargs=criterion_kwargs,
                           model_type=FAST_DRO_MODEL,
                           **model_kwargs)

    elif kind == GBM_MODEL:
        model = GradientBoostingClassifier(**model_kwargs)

    elif kind == GROUP_DRO_MODEL:
        assert criterion_kwargs[
                   "criterion_name"] == src.torchutils.criterion.GROUP_DRO_CRITERION
        group_weights_step_size = criterion_kwargs.pop(
            "group_weights_step_size")

        model = GroupDROModel(
            d_in=d, device=device, is_regressor=False,
            n_groups=dset.n_groups,
            group_weights_step_size=group_weights_step_size,
            criterion_kwargs=criterion_kwargs,
            model_type=GROUP_DRO_MODEL,
            **model_kwargs)

    elif kind == HISTGBM_MODEL:
        model = HistGradientBoostingClassifier(**model_kwargs)

    elif kind == IWC_MODEL:
        model = WeightedCovariateShiftClassifier(**model_kwargs)

    elif kind == L2_MODEL:
        model = linear_model.LogisticRegression(**model_kwargs)

    elif kind == LFR_PREPROCESSOR_MODEL:
        assert (not dset.make_dummies)
        assert dset.label_encode_categorical_cols
        lfr = LFR(unprivileged_groups=unprivileged_groups,
                  privileged_groups=privileged_groups,
                  verbose=0, **criterion_kwargs)
        base_learner_kind = model_kwargs.pop("base_learner")
        base_learner_kwargs = model_kwargs.pop("base_learner_kwargs")
        base_learner = get_classifier(dset, base_learner_kind,
                                      model_kwargs=base_learner_kwargs)
        model = ModelWithLFRPreprocessor(preprocessor=lfr,
                                         base_model=base_learner,
                                         protected_attribute_names=dset.sens,
                                         **model_kwargs)

    elif kind == LIGHTGBM_MODEL:
        model = LGBMClassifier(**model_kwargs)

    elif kind == LR_MODEL:
        model = linear_model.LogisticRegression(max_iter=1000, **model_kwargs)
    elif kind == MARGINAL_DRO_MODEL:
        assert criterion_kwargs["criterion_name"] == \
               src.torchutils.criterion.MARGINAL_DRO_CRITERION
        assert isinstance(dset, TabularDataset)
        criterion_kwargs["x_in"] = dset.X_tr[dset.sens]
        model = DenseModelWithLossParams(d_in=d, device=device,
                                         is_regressor=False,
                                         criterion_kwargs=criterion_kwargs,
                                         model_type=MARGINAL_DRO_MODEL,
                                         **model_kwargs)
    elif kind == MWLD_MODEL:
        assert criterion_kwargs["criterion_name"] in (
            src.torchutils.criterion.LVR_CRITERION,
            src.torchutils.criterion.CLV_CRITERION)
        model = DenseModel(d, device,
                           is_regressor=False,
                           criterion_kwargs=criterion_kwargs,
                           model_type=MWLD_MODEL,
                           **model_kwargs)

    elif kind == MLP_MODEL:
        model = DenseModel(d, device, is_regressor=False,
                           criterion_kwargs=criterion_kwargs,
                           **model_kwargs)

    elif kind == POSTPROCESSOR_MODEL:

        base_learner_kind = model_kwargs.pop("base_learner")
        base_learner_kwargs = model_kwargs.pop("base_learner_kwargs")
        base_learner = get_classifier(dset, base_learner_kind,
                                      model_kwargs=base_learner_kwargs)

        constraint_type = model_kwargs.pop("postprocessor_constraint")

        if constraint_type == EO_CONSTRAINT:
            postprocessor = EqOddsPostprocessing(
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups,
                **criterion_kwargs)
            model = ModelWithPostprocessor(postprocessor, base_learner,
                                           protected_attribute_names=dset.sens,
                                           **model_kwargs)
        elif constraint_type == DP_CONSTRAINT:
            model = CustomThresholdOptimizer(estimator=base_learner,
                                             constraints="demographic_parity",
                                             sensitive_features=dset.sens,
                                             **model_kwargs)

    elif kind == RANDOM_FOREST_MODEL:
        model = RandomForestClassifier(**model_kwargs)

    elif kind == SVM_MODEL:
        kernel_kwargs = model_kwargs.pop("kernel_kwargs")
        kernel_fn = model_kwargs.pop("kernel_fn")
        preprocessor = kernel_fn(**kernel_kwargs)
        # We use sklearn.svm.SVC, not sklearn.svm.LinearSVC, because the latter
        # has no .predict_proba() method.
        base_model = sklearn.svm.LinearSVC(**model_kwargs, dual=False)
        model = SVMWithKernelApprox(preprocessor, base_model)

    elif kind == XGBOOST_MODEL:
        import xgboost as xgb
        if device.type == 'cuda':
            model_kwargs['gpu_id'] = 0
            model_kwargs['tree_method'] = 'gpu_hist'
        else:
            model_kwargs['tree_method'] = 'hist'
        model = xgb.XGBClassifier(**model_kwargs)
    else:
        raise NotImplementedError(f"classifier type {kind} is not implemented.")
    return model


def fit_model(dset: Dataset,
              model,
              sample_weights=None,
              **fit_kwargs):
    """Fit a model with the specified parameters."""
    logging.info(f"fitting model {type(model)} with fit_kwargs {fit_kwargs}")
    X_tr, y_tr, X_te, y_te, X_val, y_val = dset.get_data()
    logging.info("fitting model with training data shape {}".format(X_tr.shape))
    if "is_regression" in fit_kwargs:
        fit_kwargs.pop("is_regression")

    # Model training
    if isinstance(model, DenseModel):
        # get loaders and fit the model
        batch_size = fit_kwargs.pop("batch_size")
        loader = dset.get_dataloader("train", batch_size=batch_size)
        model.fit(loader, X_val=X_val, y_val=y_val,
                  sensitive_features=dset.sens,
                  **fit_kwargs)

    elif fit_kwargs:
        model.fit(X_tr, y_tr, sample_weight=sample_weights, **fit_kwargs)

    else:
        model.fit(X_tr, y_tr, sample_weight=sample_weights)

    # Model evaluation
    if X_val is not None and y_val is not None:
        log_eval_metrics(model, X=X_te, y=y_te,
                         sensitive_features=dset.sens,
                         is_regression=dset.is_regression,
                         suffix='test')
        log_eval_metrics(model, X=X_val, y=y_val,
                         sensitive_features=dset.sens,
                         is_regression=dset.is_regression,
                         suffix='val')
    else:
        loader = dset.get_dataloader("test")
        log_eval_metrics(model, loader=loader,
                         is_regression=dset.is_regression,
                         suffix="test")
        loader = dset.get_dataloader("validation")
        log_eval_metrics(model, loader=loader,
                         is_regression=dset.is_regression,
                         suffix="val")
    logging.info('model fitting complete')
    return model


def get_model(dset: TabularDataset, kind: str, criterion_kwargs: dict,
              model_kwargs: dict, device=None):
    """Instantiate and return a model with the specified parameters."""
    model = get_classifier(dset=dset, kind=kind, device=device,
                           criterion_kwargs=criterion_kwargs,
                           model_kwargs=model_kwargs)
    return model


def get_and_fit_model(dset: TabularDataset,
                      kind: str,
                      model_kwargs: dict,
                      criterion_kwargs: dict,
                      opt_kwargs: dict,
                      optimizer: Optional[str] = None,
                      device=None,
                      **fit_kwargs):
    """Fetch a model and train it."""
    logging.info(f"fitting {kind} model with optimizer {optimizer}")

    model = get_model(dset=dset, kind=kind,
                      criterion_kwargs=criterion_kwargs,
                      model_kwargs=model_kwargs,
                      device=device)

    if opt_kwargs:
        opt = src.torchutils.get_optimizer(optimizer,
                                           model, **opt_kwargs)
        fit_kwargs.update({"optimizer": opt})

    model = fit_model(dset, is_regression=dset.is_regression, model=model,
                      **fit_kwargs)

    return model
