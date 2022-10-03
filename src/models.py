"""
Custom model classes, e.g. to provide sklearn-style interfaces.
"""

from abc import ABC, abstractmethod
import logging
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing import LFR

from fairlearn.reductions import ExponentiatedGradient
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn import ensemble, svm, kernel_approximation, linear_model
import fairlearn

assert fairlearn.__version__.split('.')[1] == '7'

from src.aif360_utils import to_binary_label_dataset, to_structured_dataset
from src.utils import LOG_LEVEL

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


class CustomExponentiatedGradient(ExponentiatedGradient):
    """Custom class to allow for scikit-learn-compatible interface.

    Specifically, this method takes (and ignores) a sample_weights
    parameter to its .fit() method; otherwise identical to
    fairlearn.ExponentiatedGradient.
    """

    def __init__(self, sensitive_features: List[str], **kwargs):
        super().__init__(**kwargs)
        self.sensitive_features = sensitive_features

    def fit(self, X, y, sample_weight=None, **kwargs):
        del sample_weight
        super().fit(X.values, y.values,
                    sensitive_features=X[self.sensitive_features].values,
                    **kwargs)

    def predict_proba(self, X):
        """Alias to _pmf_predict(). Note that this tends to return 'hard'
        predictions, which don't perform well for metrics like cross-entropy."""
        return super()._pmf_predict(X)


class CustomThresholdOptimizer(ThresholdOptimizer):
    def __init__(self, sensitive_features: List[str], **kwargs):
        super().__init__(**kwargs, predict_method='predict')
        self.sensitive_features = sensitive_features

    def fit(self, X, y, sample_weight=None):
        del sample_weight
        super().fit(X, y, sensitive_features=X[self.sensitive_features])

    def predict(self, X, random_state=None):
        return super().predict(X, sensitive_features=X[self.sensitive_features],
                               random_state=random_state)

    def predict_proba(self, X):
        return super()._pmf_predict(
            X, sensitive_features=X[self.sensitive_features])


class NoClipSquareLoss:
    """Class to evaluate the square loss without clipping.

    Compare to class fairlearn.reductions.SquareLoss (which uses clipping).
    """

    def __init__(self):
        return

    def eval(self, y_true, y_pred):  # noqa: A003
        """Evaluate the square loss for the given set of true and predicted values."""
        return (y_true - y_pred) ** 2


class ModelWithPostprocessor(ensemble.GradientBoostingClassifier):
    def __init__(self, postprocessor: EqOddsPostprocessing,
                 base_model: Union[
                     ensemble.GradientBoostingClassifier,
                     ensemble.GradientBoostingClassifier],
                 protected_attribute_names: List[str],
                 **kwargs):
        if 'base_learner' in kwargs:
            del kwargs['base_learner']
        self.base_model = base_model
        self.postprocessor = postprocessor
        self.protected_attribute_names = tuple(protected_attribute_names)

    def fit(self, X, y, **kwargs):
        # Fit the model
        self.base_model.fit(X, y, **kwargs)
        # Fit the postprocessor
        y_pred = self.base_model.predict(X)
        y_pred = pd.Series(y_pred, name="target")
        dataset_true = to_binary_label_dataset(
            pd.concat((X, y), axis=1),
            protected_attribute_names=self.protected_attribute_names)
        dataset_pred = to_binary_label_dataset(
            pd.concat((X, y_pred), axis=1),
            protected_attribute_names=self.protected_attribute_names)
        self.postprocessor.fit(dataset_true=dataset_true,
                               dataset_pred=dataset_pred)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        y_pred = self.base_model.predict(X)
        y_pred = pd.Series(y_pred, name="target")
        dataset_pred = to_binary_label_dataset(
            pd.concat((X.set_index(y_pred.index), y_pred), axis=1),
            protected_attribute_names=self.protected_attribute_names)
        y_pred_transformed = self.postprocessor.predict(dataset_pred)
        return y_pred_transformed.labels.ravel()


class ModelWithPreprocessor(ABC):
    def __init__(self, preprocessor, base_model, **kwargs):
        self.preprocessor = preprocessor
        self.base_model = base_model

    @abstractmethod
    def preprocess(self, X, y):
        raise

    def predict(self, X):
        return self.base_model.predict(X)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    @abstractmethod
    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit the preprocessor and the base model."""
        raise


class ModelWithLFRPreprocessor(ModelWithPreprocessor):
    def __init__(self, preprocessor: LFR, base_model: Union[
        ensemble.GradientBoostingClassifier,
        ensemble.GradientBoostingClassifier],
                 protected_attribute_names: List[str],
                 maxiter,
                 maxfun,
                 **kwargs):
        super().__init__(preprocessor, base_model)
        if 'base_learner' in kwargs:
            del kwargs['base_learner']
        self.maxiter = maxiter
        self.maxfun = maxfun
        self.protected_attribute_names = tuple(protected_attribute_names)

    def preprocess(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = pd.concat((X, y), axis=1)
        dataset = to_structured_dataset(
            df, protected_attribute_names=self.protected_attribute_names)
        transformed = self.preprocessor.transform(dataset)
        X_preproc = pd.DataFrame(transformed.features,
                                 columns=X.columns)
        # y_preproc = pd.Series(transformed.labels.ravel(), name=y.name)
        return X_preproc

    def fit(self, X, y, sample_weight=None, **kwargs):
        if not kwargs:
            kwargs = {}
        df = pd.concat((X, y), axis=1)

        dataset = to_structured_dataset(
            df, protected_attribute_names=self.protected_attribute_names)

        # Fit the preprocessor and obtain the "fair" representation
        # of the training data
        logging.info("fitting preprocessor with %s  samples", len(X))
        self.preprocessor.fit(dataset,
                              maxiter=self.maxiter, maxfun=self.maxfun)
        X_transformed = self.preprocess(X, y)
        logging.info("preprocessor fitting complete; training model")
        self.base_model.fit(X_transformed, y, sample_weight=sample_weight,
                            **kwargs)
        return


class SVMWithKernelApprox(ModelWithPreprocessor):
    def __init__(self,
                 preprocessor: Union[kernel_approximation.Nystroem,
                                     kernel_approximation.RBFSampler],
                 base_model: Union[svm.LinearSVC, svm.SVC]):
        super().__init__(preprocessor, base_model)

    def preprocess(self, X, y):
        return self.preprocessor.transform(X)

    def predict_proba(self, X):
        logging.warn("calling SVMWithKernelApprox.predict_proba()"
                     "gives only 'hard' predictions like .predict()"
                     "(there is no LinearSVC.predict_proba() method to call!)")
        return self.base_model.predict(X)

    def fit(self, X, y, sample_weight=None, **kwargs):
        # fit the preprocessor
        self.preprocessor.fit(X, y)
        logging.info("fitting preprocessor with %s  samples", len(X))
        X_transformed = self.preprocess(X, y=None)
        logging.info("preprocessor fitting complete; training model")
        self.base_model.fit(X=X_transformed, y=y, sample_weight=sample_weight)
        return


class WeightedCovariateShiftClassifier:
    def __init__(self):
        # Used to predict weights for training examples
        self.domain_classifier = linear_model.LogisticRegression()
        # Used to predict labels
        self.discriminator = linear_model.LogisticRegression()

    def predict_importance_weights(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        weights = X @ np.squeeze(self.domain_classifier.coef_, 0)
        return weights

    def fit(self, X_id, y_id, X_ood, y_ood):
        # Fit the domain classifier
        if isinstance(X_id, pd.DataFrame):
            X = pd.concat((X_id, X_ood), axis=0)
        else:
            X = np.row_stack((X_id, X_ood))
        y = np.concatenate((np.ones_like(y_id), -np.ones_like(y_ood)))
        logging.info("fitting domain classifier")
        self.domain_classifier.fit(X, y)

        # Fit the discriminator
        logging.info("fitting discriminator")
        id_sample_weights = self.predict_importance_weights(X_id)
        self.discriminator.fit(X_id, y_id, sample_weight=id_sample_weights)

    def predict(self, X):
        return self.discriminator.predict(X)

    def predict_proba(self, X):
        return self.discriminator.predict_proba(X)
