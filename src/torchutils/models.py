"""
Torch models with sklearn-style interfaces.
"""

import copy
import functools
import logging
import math
import os
from tempfile import TemporaryDirectory
import time
from typing import Callable, List, Union

import numpy as np
import pandas as pd
from scipy.optimize import brent
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.functional import binary_cross_entropy, mse_loss
from torchinfo import summary
from tqdm import tqdm
import wandb

from src.torchutils import pd_to_torch_float
from src.torchutils.metrics import binary_accuracy_from_probs
from src.torchutils.criterion import get_criterion
from src.marginal_dro.dual_lip_risk_bound import LipLoss, opt_model
from src.marginal_dro.utils import calc_rho
from src.metrics import metrics_by_group
from src.utils import LOG_LEVEL, make_uid

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def get_steps(steps, epochs, train_loader, batch_size=None):
    """Helper function to compute number of training steps.

    If provided, steps overrides epochs.
    """
    n_train = len(train_loader.dataset)
    if not batch_size:
        # Note: Accelerator removes the batch size attribute.
        assert train_loader.batch_size
        batch_size = train_loader.batch_size
    if steps and epochs:
        # Case: steps and epochs both specified; default to steps.
        logging.warning(
            "steps={} and epochs={} defined; defaulting to steps".format(
                steps, epochs))
    elif epochs:
        # Case: only epochs specified. Compute the number of train steps.
        steps = math.ceil(n_train / batch_size) * epochs
        logging.info(
            "training for {} steps <=> {} epochs with n_tr={}, b={}".format(
                steps, epochs, n_train, batch_size))
    return steps


class SklearnStylePytorchModel(nn.Module):
    """A pytorch model with an sklearn-style interface."""

    def __init__(self):
        super().__init__()

    def init_layers(self):
        raise

    def forward(self, X):
        raise

    def predict(self, X) -> np.ndarray:
        raise

    def predict_proba(self, X) -> np.ndarray:
        raise

    def print_summary(self, global_step: int, metrics):
        # Print a summary every n steps
        if (global_step % 100) == 0:
            metrics_str = ', '.join(
                [f"{k}: {v}" for k, v in sorted(metrics.items())])
            logging.info(
                "metrics for model {} at step {}: {}".format(
                    self.model_type, global_step, metrics_str))

    def disparity_metric_fn(self) -> Callable:
        raise

    def _check_inputs(self, X, y):
        raise

    def _compute_validation_metrics(self, X_val,
                                    y_val, sens_val) -> dict:
        raise

    def _update(self, optimizer, X, y, g):
        """Execute a single parameter update step."""
        raise

    def fit(self, train_loader, X_val, y_val,
            sensitive_features: List[str],
            optimizer, steps: int = None,
            epochs: int = None,
            scheduler=None,
            verbose=0,
            sample_weight=None):
        raise


class DenseModel(SklearnStylePytorchModel):
    """A scikit-learn style interface for a pytorch model."""

    def __init__(self, d_in: int, device: str,
                 is_regressor: bool,
                 criterion_kwargs: dict,
                 d_hidden: int,
                 dropout_prob=None,
                 num_layers: int = 1,
                 model_type: str = "default",
                 max_samples_training_val: int = 4096,
                 tmp_file_dir="./tmp"):
        super(DenseModel, self).__init__()
        self.device = device
        self.is_regressor = is_regressor
        criterion_name = criterion_kwargs.pop("criterion_name")
        self.criterion_name = criterion_name
        self.criterion_kwargs = criterion_kwargs
        self.criterion = get_criterion(self.criterion_name,
                                       self.is_regressor,
                                       device=self.device,
                                       **self.criterion_kwargs)
        # During training, limit the validation set to at most this size.
        self.max_samples_training_val = max_samples_training_val
        self.uid = make_uid()
        self.tmp_file_dir = tmp_file_dir
        if not os.path.exists(tmp_file_dir):
            os.makedirs(tmp_file_dir)
        logging.info("generated model uid %s" % self.uid)

        if not d_hidden:
            self.d_hidden = d_in
        else:
            self.d_hidden = d_hidden

        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.d_in = d_in  # input feature dimension

        self.layers = nn.ModuleList()
        self.init_layers()
        self.model_type = model_type  # used for logging
        self.to(self.device)

        # print a model summary w/batch size 128, as sanity check
        summary(self, input_size=(128, self.d_in))

    def init_layers(self):
        # Initialize the model layers.
        d_in = self.d_in
        for i in range(self.num_layers + 1):
            if i == self.num_layers:
                d_out = 1
            else:
                d_out = self.d_hidden
            self.layers.append(nn.Linear(d_in, d_out))
            if self.dropout_prob:
                self.layers.append(nn.Dropout(p=self.dropout_prob))
            d_in = d_out

        self.to(self.device)

        # print a model summary w/batch size 128, as sanity check
        summary(self, input_size=(128, self.d_in))

    def forward(self, x):
        x = x.to(self.device)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.squeeze(x)  # [batch_size,1] --> batch_size,]

            # Apply relu at all non-terminal layers, after dropout
            if i != len(self.layers) and \
                    (isinstance(layer, torch.nn.Dropout)
                     or not self.dropout_prob):
                x = torch.relu(x)

        # Sigmoid activation for output of classifier only; for regressor
        # the final layer is strictly linear (no activation).

        if not self.is_regressor:
            x = torch.sigmoid(x)
        return x

    def predict_proba(self, X) -> np.ndarray:
        """'Soft' prediction function analogous to sklearn model.predict()."""
        self.eval()
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = pd_to_torch_float(X)
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs.detach().cpu().numpy()

    def predict(self, x) -> np.ndarray:
        """'Hard' prediction function analogous to sklearn model.predict()."""
        self.eval()
        outputs = self.predict_proba(x)
        # Binarize the outputs
        return (outputs > 0.5).astype(int)

    @property
    def _disparity_metric_fn(self) -> Callable:
        if self.is_regressor:
            raise NotImplementedError
        else:
            return functools.partial(metrics_by_group, suffix="training_val")

    def _check_inputs(self, X: np.ndarray, y: np.ndarray):
        assert not np.any(np.isnan(X)), "null inputs"
        assert not np.any(np.isnan(y)), "null targets"
        if not self.is_regressor:
            assert np.all(np.isin(y, (0., 1.))), "nonbinary labels"

    def _compute_validation_metrics(self, X_val,
                                    y_val, sens_val: pd.DataFrame) -> dict:
        with torch.no_grad():
            outputs_val = self.forward(X_val)

            if torch.any(torch.isnan(outputs_val)):
                return {}
            yhats_soft = outputs_val.detach().cpu().numpy()
            log_metrics = self._disparity_metric_fn(
                y_true=y_val.detach().cpu().numpy(),
                yhat_soft=yhats_soft,
                yhat_hard=(yhats_soft > 0.5).astype(float),
                sensitive_features=sens_val)

            log_metrics["training_val_loss"] = self.criterion(outputs_val,
                                                              y_val).item()

            if self.is_regressor:
                log_metrics["training_val_mse"] = mse_loss(
                    outputs_val, y_val, reduction="mean").item()
            else:
                log_metrics["training_val_ce"] = binary_cross_entropy(
                    outputs_val,
                    y_val).item()
                log_metrics[
                    "training_val_accuracy"] = binary_accuracy_from_probs(
                    inputs=outputs_val, labels=y_val)

        return log_metrics

    def _update(self, optimizer, X, y, g):
        """Execute a single parameter update step."""
        self.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = self.forward(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        optimizer.step()
        return loss, outputs

    def save_weights(self, dirname):
        weights_fp = os.path.join(dirname, 'weights.pt')
        logging.info("saving weights to %s" % weights_fp)
        torch.save(self.state_dict(), weights_fp)
        return

    def load_weights(self, dirname):
        weights_fp = os.path.join(dirname, 'weights.pt')
        logging.info("loading weights from %s" % weights_fp)
        state_dict = torch.load(weights_fp)
        self.load_state_dict(state_dict)
        return

    def fit(self, train_loader, X_val, y_val,
            sensitive_features: List[str],
            optimizer, steps: int = None,
            epochs: int = None,
            scheduler=None,
            verbose=0,
            log_freq=None,
            keep_best_per_epoch=True,
            sample_weight=None):
        self.train()
        del sample_weight
        steps = get_steps(steps, epochs, train_loader)

        if len(X_val) > self.max_samples_training_val:
            logging.info("downsampling training_val set from "
                         f"{len(X_val)} to {self.max_samples_training_val}")
            idxs = np.random.choice(len(X_val),
                                    size=self.max_samples_training_val,
                                    replace=False)
            X_val = X_val.loc[idxs]
            y_val = y_val.loc[idxs]
        sens_val = X_val[sensitive_features]
        X_val = pd_to_torch_float(X_val).to(self.device)
        y_val = pd_to_torch_float(y_val).to(self.device)

        logging.info("training pytorch model type %s for %s steps.",
                     self.model_type, steps)

        epoch = 0
        best_epoch = 0
        global_step = 0
        best_epoch_loss = float('inf')

        with TemporaryDirectory(dir=self.tmp_file_dir) as train_dir, tqdm(
                total=steps) as pbar:
            self.save_weights(train_dir)
            while True:
                loader = train_loader
                for batch_idx, batch in enumerate(loader):

                    data, labels, sens = batch
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    sens = sens.to(self.device)

                    # With batch of one observation, updates can be unstable
                    # due to large targets/loss values; skip these updates.
                    if len(data) == 1:
                        continue

                    # Model update step
                    train_loss, _ = self._update(optimizer, data, labels, sens)

                    # Kill training if loss is nan.
                    if torch.any(torch.isnan(train_loss)):
                        logging.error("NaN loss detected; terminating training"
                                      "{} at epoch {} step {} global_step {}"
                                      .format(self.model_type, epoch, batch_idx,
                                              global_step))
                        self.load_weights(train_dir)
                        return

                    pbar.update(n=1)

                    # Schedule happens on per-step, not per-epoch, basis.
                    if scheduler is not None: scheduler.step()

                    global_step += 1

                    if global_step >= steps:
                        break

                # Epoch end (or train end); compute and log train/val metrics.
                logging.info(f"epoch {epoch} end.")
                val_metrics = self._compute_validation_metrics(
                    X_val=X_val, y_val=y_val, sens_val=sens_val)
                train_loss = train_loss.item()
                val_metrics["train_loss"] = train_loss
                val_loss = val_metrics["training_val_loss"]
                wandb.log(val_metrics)

                pbar.set_description(
                    "epoch {} train_loss: {} val_loss: {}".format(
                        epoch,
                        round(train_loss, 2),
                        round(val_metrics["training_val_loss"], 2)))

                # Weight saving
                if val_loss < best_epoch_loss and keep_best_per_epoch:
                    # Case: only save best epoch weights; this is current best
                    logging.info(
                        "loss at epoch {} {} < best of {}; saving".format(
                            epoch, val_loss, best_epoch_loss))
                    best_epoch_loss = val_loss
                    best_epoch = epoch
                    self.save_weights(train_dir)
                elif not keep_best_per_epoch:
                    # Case: always save the weights (not only on best epoch)
                    self.save_weights(train_dir)
                else:
                    # Case: only save best epoch weights; this is not best.
                    logging.info(
                        "loss at epoch {} {} > best of {}; not saving".format(
                            epoch, val_loss, best_epoch_loss))
                epoch += 1

                if global_step >= steps:
                    # Training terminated; the best weights
                    self.load_weights(train_dir)
                    wandb.log({"best_epoch": best_epoch})
                    return


class DenseModelWithLossParams(DenseModel):
    def __init__(self, p_min: float, niter_inner: int, nbisect: int, **kwargs):
        super().__init__(**kwargs)
        self.eta_train = None
        self.niter_inner = niter_inner
        self.p_min = p_min
        self.nbisect = nbisect

        # Validation robust loss setup
        self.criterion_val = None  # This is initialized at fit time.
        self.eta_val = None

    @property
    def rho(self):
        return calc_rho(self.p_min)

    def _compute_validation_metrics(self, X_val,
                                    y_val, sens_val: pd.DataFrame) -> dict:
        assert isinstance(self.criterion_val, LipLoss)
        assert self.eta_val is not None

        with torch.no_grad():
            outputs_val = self.forward(X_val)

            if torch.any(torch.isnan(outputs_val)):
                return {}
            yhats_soft = outputs_val.detach().cpu().numpy()
            log_metrics = self._disparity_metric_fn(
                y_true=y_val.detach().cpu().numpy(),
                yhat_soft=yhats_soft,
                yhat_hard=(yhats_soft > 0.5).astype(float),
                sensitive_features=sens_val)

            val_bce_loss = nn.BCELoss(reduction="none")
            val_per_ex_loss = val_bce_loss.forward(outputs_val, y_val)
            val_robust_loss = self.criterion_val.forward(val_per_ex_loss,
                                                         self.eta_val)
            log_metrics["training_val_loss"] = val_robust_loss.item()

            if self.is_regressor:
                log_metrics["training_val_mse"] = mse_loss(
                    outputs_val, y_val, reduction="mean").item()
            else:
                log_metrics["training_val_ce"] = binary_cross_entropy(
                    outputs_val,
                    y_val).item()
                log_metrics[
                    "training_val_accuracy"] = binary_accuracy_from_probs(
                    inputs=outputs_val, labels=y_val)

        return log_metrics

    def _update(self, optimizer, X, y, g):
        """Execute a single parameter update step."""
        loss, outputs = super()._update(optimizer, X, y, g)
        if isinstance(self.criterion, LipLoss):
            self.criterion.project()
        else:
            raise NotImplementedError(
                "the class DenseModelWithLossParams only supports LipLoss; "
                f"if your loss of type {type(self.criterion)} has trainable "
                f"parameters, it needs to be added to the implementation.")
        return loss, outputs

    def _init_criterion(self, x_in):
        criterion_kwargs = copy.copy(self.criterion_kwargs)
        criterion_kwargs["x_in"] = x_in
        criterion = get_criterion(self.criterion_name,
                                  self.is_regressor,
                                  device=self.device,
                                  **criterion_kwargs)
        return criterion

    def _init_eta(self, x_in, y_in, lr: float, criterion):
        """Initialize the dual variable eta.

        See marginal_dro.dual_lip_risk_bound.opt_model().
        """
        assert isinstance(criterion, LipLoss)
        loss = nn.BCELoss(reduction="none")

        wrapped_fun = lambda eta: opt_model(self, loss, criterion, 0.0,
                                            self.rho, x_in=x_in, y_in=y_in,
                                            lr=lr,
                                            niter=self.niter_inner)[0][-1]
        opt_init = opt_model(self, loss, criterion, 0.0, self.rho, x_in,
                             y_in, lr=lr, niter=self.niter_inner)
        brack_ivt = (min(0, np.nanmin(opt_init[1])), np.nanmax(opt_init[1]))
        bopt = brent(wrapped_fun, brack=brack_ivt, maxiter=self.nbisect,
                     full_output=True)
        eta = bopt[0]
        return eta

    def fit(self, train_loader, X_val, y_val,
            sensitive_features: List[str],
            optimizer, steps: int = None,
            epochs: int = None,
            scheduler=None,
            verbose=0,
            log_freq=None,
            keep_best_per_epoch=True,
            sample_weight=None):

        lr = optimizer.param_groups[0]['lr']

        # Initialize eta (the dual variable) on the train data.
        data, labels, sens = next(iter(train_loader))
        logging.info("initializing training eta...")
        self.eta_train = self._init_eta(x_in=data, y_in=labels,
                                        lr=lr, criterion=self.criterion)
        logging.info("initializing training eta complete.")

        # Initialize eta on the validation data, so we can compute the
        # validation loss.
        logging.info("initializing validation eta...")
        self.criterion_val = self._init_criterion(
            X_val[sensitive_features].values)
        self.eta_val = self._init_eta(x_in=X_val.values, y_in=y_val.values,
                                      lr=lr, criterion=self.criterion_val)
        logging.info("initializing validation eta complete.")

        super().fit(train_loader=train_loader, X_val=X_val, y_val=y_val,
                    sensitive_features=sensitive_features, optimizer=optimizer,
                    steps=steps, epochs=epochs, scheduler=scheduler,
                    verbose=verbose, log_freq=log_freq,
                    keep_best_per_epoch=keep_best_per_epoch,
                    sample_weight=sample_weight)


class GroupDROModel(DenseModel):
    def __init__(self, group_weights_step_size, n_groups=2, **kwargs):
        super().__init__(**kwargs)
        self.group_weights_step_size = torch.Tensor(
            [group_weights_step_size]).to(self.device)
        # initialize adversarial weights
        self.group_weights = torch.ones(n_groups, device=self.device)
        self.group_weights = self.group_weights / self.group_weights.sum()

    def _compute_validation_metrics(self, X_val,
                                    y_val, sens_val) -> dict:
        log_metrics = {}

        with torch.no_grad():
            outputs_val = self.forward(X_val)

            if torch.any(torch.isnan(outputs_val)):
                return {}
            group_losses = self.criterion(outputs_val, y_val,
                                          pd_to_torch_float(sens_val))
            val_loss = group_losses @ self.group_weights
            val_ce = binary_cross_entropy(outputs_val, y_val, reduction="mean")
            yhats_soft = outputs_val.detach().cpu().numpy()
            disparity_metrics = self._disparity_metric_fn(
                y_true=y_val.detach().cpu().numpy(),
                yhat_soft=yhats_soft,
                yhat_hard=(yhats_soft > 0.5).astype(float),
                sensitive_features=sens_val)
            if self.is_regressor:
                log_metrics["training_val_mse"] = mse_loss(
                    outputs_val, y_val, reduction="mean").item()
            else:
                log_metrics["training_val_ce"] = binary_cross_entropy(
                    outputs_val, y_val).item()
                log_metrics[
                    "training_val_accuracy"] = binary_accuracy_from_probs(
                    inputs=outputs_val, labels=y_val)
        return {"training_val_loss": val_loss.item(),
                "training_val_ce": val_ce.item(),
                **log_metrics,
                **disparity_metrics, }

    def _update(self, optimizer, X, y, g):
        """Execute a single parameter update step."""
        self.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = self.forward(X)

        # send sens. to device; it is not stored on GPU for other methods bc
        # it is not needed for loss computation.
        g = g.to(self.device)

        group_losses = self.criterion(outputs, y, g)
        # update group weights
        self.group_weights = self.group_weights * torch.exp(
            self.group_weights_step_size * group_losses.data)
        self.group_weights = (self.group_weights / (self.group_weights.sum()))
        # update model
        loss = group_losses @ self.group_weights
        loss.backward()
        optimizer.step()
        return loss, outputs

