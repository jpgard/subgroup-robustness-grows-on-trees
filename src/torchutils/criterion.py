"""
Training criteria for use with torch modules.
"""

import functools
import itertools
import logging
import math
from typing import Callable

import torch
from scipy import optimize as sopt
from torch.nn import functional as F
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy

from src.fastdro.robust_losses import RobustLoss
from src.marginal_dro.dual_lip_risk_bound import LipLoss
from src.utils import LOG_LEVEL

MSE_CRITERION = "mse"
CE_CRITERION = "ce"
FASTDRO_CRITERION = "fastdro"
LVR_CRITERION = "loss_variance"
CLV_CRITERION = "coarse_loss_variance"
DORO_CRITERION = "doro"
GROUP_DRO_CRITERION = "groupdro"
DEEP_LFR_CRITERION = "deeplfr"
MARGINAL_DRO_CRITERION = "marginaldro"

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


class DeepLFRCriterion(Module):
    def __init__(self, Ax, Ay, Az):
        super().__init__()
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        # L_z, L_y, L_z store the most recent value of each loss component;
        # useful for logging and because we do not want to change the return
        # signature of the loss criterion for consistency across losses.
        self.L_x = None
        self.L_y = None
        self.L_z = None

    def __call__(self, P_z, X_hat, X, g, y, y_hat):
        # L_z; ensures mapping from X_0 to Z satisfies statistical parity

        # g has shape num_sens, num_attrs
        # Higher-dimensional generalization of L_z from original paper.
        # Instead of |M_plus - M_minus|, we compute max_{i,j}|M_i - M_j|
        M_by_subgroup = []
        n_attrs = g.shape[1]
        for subgroup_idxs in itertools.product(*[(0, 1)] * n_attrs):
            subgroup_idxs = torch.Tensor(subgroup_idxs).to(g.device)
            mask = torch.all(g == subgroup_idxs, dim=1)
            M_subgroup = torch.sum(P_z[mask], axis=0)  # shape [k,]
            M_by_subgroup.append(M_subgroup)
        M_by_subgroup = torch.stack(M_by_subgroup)  # shape [n_groups, k]
        M_plus, _ = torch.max(M_by_subgroup, dim=0)  # shape [k,]
        M_minus, _ = torch.min(M_by_subgroup, dim=0)  # shape [k,]

        self.L_z = torch.sum(torch.abs(M_plus - M_minus))

        # L_x; ensures x_hat and x do not change too much in L2-distance
        self.L_x = torch.sum(torch.linalg.norm(X - X_hat, dim=1, ord=2) ** 2)

        # L_y; prediction error
        self.L_y = binary_cross_entropy(input=y_hat, target=y, reduction='sum')
        return self.Ax * self.L_x + self.Ay * self.L_y + self.Az * self.L_z


# DORO; adapted from
# https://github.com/RuntianZ/doro/blob/master/wilds-exp/algorithms/doro.py
def chi_square_doro_criterion(outputs, targets, alpha, eps):
    batch_size = len(targets)
    loss = binary_cross_entropy(outputs, targets, reduction="none")
    # Chi^2-DORO
    max_l = 10.
    C = math.sqrt(1 + (1 / alpha - 1) ** 2)
    n = int(eps * batch_size)
    rk = torch.argsort(loss, descending=True)
    l0 = loss[rk[n:]]
    foo = lambda eta: C * math.sqrt(
        (F.relu(l0 - eta) ** 2).mean().item()) + eta
    opt_eta = sopt.brent(foo, brack=(0, max_l))
    loss = C * torch.sqrt((F.relu(l0 - opt_eta) ** 2).mean()) + opt_eta
    return loss


# DORO; adapted from
# https://github.com/RuntianZ/doro/blob/master/wilds-exp/algorithms/doro.py
def cvar_doro_criterion(outputs, targets, eps, alpha):
    batch_size = len(targets)
    loss = binary_cross_entropy(outputs, targets, reduction="none")
    # CVaR-DORO
    gamma = eps + alpha * (1 - eps)
    n1 = int(gamma * batch_size)
    n2 = int(eps * batch_size)
    rk = torch.argsort(loss, descending=True)
    loss = loss[rk[n2:n1]].sum() / alpha / (batch_size - n2)
    return loss


def marginal_dro_criterion(
        outputs, targets, sens, radius, b_init, k_dual, eta):
    lip_loss = LipLoss(radius=radius, x_in=sens, b_init=b_init, k_dual=k_dual,
                       eta=eta)
    elementwise_loss = binary_cross_entropy(input=outputs,
                                            target=targets,
                                            reduction="none")
    loss = lip_loss(losses=elementwise_loss, eta=eta)
    return loss


def get_criterion(criterion_name: str, is_regressor: bool, device,
                  **kwargs) -> Callable:
    logging.info(
        "Received the following criterion parameters: "
        f"criterion_name {criterion_name} is_regressor {is_regressor} {kwargs}")

    if criterion_name == MSE_CRITERION and is_regressor:
        return torch.nn.MSELoss()

    elif criterion_name == CE_CRITERION and (not is_regressor):
        return torch.nn.BCELoss()

    elif criterion_name == DEEP_LFR_CRITERION:
        assert not is_regressor
        return DeepLFRCriterion(**kwargs)

    elif (criterion_name == DORO_CRITERION) and (
            kwargs["geometry"] == "chi-square"):
        if is_regressor:
            raise NotImplementedError("DORO only supported for classification.")

        return functools.partial(chi_square_doro_criterion,
                                 eps=kwargs["eps"],
                                 alpha=kwargs["alpha"])

    elif (criterion_name == DORO_CRITERION) and (
            kwargs["geometry"] == "cvar"):
        if is_regressor:
            raise NotImplementedError("DORO only supported for classification.")

        return functools.partial(cvar_doro_criterion,
                                 eps=kwargs["eps"],
                                 alpha=kwargs["alpha"])

    elif criterion_name == GROUP_DRO_CRITERION:
        def _loss_fn(outputs, targets, sens):
            assert torch.all(torch.logical_or(sens == 0., sens == 1.)), \
                "only binary groups supported."
            subgroup_losses = []
            n_attrs = sens.shape[1]
            elementwise_loss = binary_cross_entropy(input=outputs,
                                                    target=targets,
                                                    reduction="none")
            # Compute the loss on each subgroup
            for subgroup_idxs in itertools.product(*[(0, 1)] * n_attrs):
                subgroup_idxs = torch.Tensor(subgroup_idxs).to(sens.device)
                mask = torch.all(sens == subgroup_idxs, dim=1)
                subgroup_loss = elementwise_loss[mask].sum() / mask.sum()
                subgroup_losses.append(subgroup_loss)
            return torch.stack(subgroup_losses)

        return _loss_fn

    elif criterion_name == MARGINAL_DRO_CRITERION:
        return LipLoss(**kwargs, eta=0.)

    elif criterion_name == FASTDRO_CRITERION:
        robust_loss = RobustLoss(
            geometry=kwargs['geometry'],
            size=float(kwargs.get('size', 1.0)),
            reg=kwargs.get('reg', 0.01),
            max_iter=kwargs.get('max_iter', 1000)
        )

        if is_regressor:
            def _loss_fn(outputs, targets):
                # taken from https://github.com/daniellevy/fast-dro/\
                # blob/dc75246ed5df5c40a54990916ec351ec2b9e0d86/train.py#L343
                return robust_loss((outputs.squeeze() - targets) ** 2)
        else:
            def _loss_fn(outputs, targets):
                return robust_loss(binary_cross_entropy(outputs, targets,
                                                        reduction="none"))
        return _loss_fn

    elif criterion_name == LVR_CRITERION:
        loss_lambda = kwargs['lv_lambda']

        def _loss_fn(outputs, targets):
            if is_regressor:
                elementwise_loss = (outputs.squeeze() - targets) ** 2
            else:
                elementwise_loss = binary_cross_entropy(outputs, targets,
                                                        reduction="none")
            loss_variance = torch.var(elementwise_loss)
            return torch.mean(elementwise_loss) + loss_lambda * loss_variance

        return _loss_fn
    else:
        raise NotImplementedError
