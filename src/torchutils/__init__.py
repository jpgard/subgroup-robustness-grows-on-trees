import logging
from typing import Union

import numpy as np
import pandas as pd
import torch
from qhoptim.pyt import QHAdam

from src.utils import LOG_LEVEL

SGD_OPT = "sgd"
ADAM_OPT = "adam"
ADAMW_OPT = "adamw"
QHADAM_OPT = "qhadam"

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    level=LOG_LEVEL,
    datefmt='%Y-%m-%d %H:%M:%S')


def get_device(disable_cuda: bool):
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def pd_to_torch_float(df: Union[pd.DataFrame, pd.Series]) -> torch.Tensor:
    return torch.from_numpy(df.values).float()


def safe_cast_to_numpy(ary):
    if isinstance(ary, np.ndarray):
        return ary
    elif isinstance(ary, torch.Tensor):
        return ary.detach().cpu().numpy()
    elif hasattr(ary, 'values'):  # covers all pandas dataframe/series types
        return ary.values
    else:
        raise NotImplementedError(f"unsupported type: {type(ary)}")


def clip_torch_outputs(t: torch.Tensor, eps=1e-8, clip_max=1.0, clip_min=0.0):
    """Helper function to safely clip tensors with values outside a range.

    This is mostly used when casting numpy arrays to torch doubles, which can
    result in values slightly outside the expected range in critical ways
    (e.g. 1. can be cast as a double to 1.0000000000000002, which raises
    errors in binary cross-entropy).
    """
    if not (t.max() <= clip_max + eps) and (t.min() >= clip_max - eps):
        logging.warning(
            f"tensor values outside clip range [{clip_min}-{eps},{clip_max}+{eps}]")
    return torch.clip(t, min=clip_min, max=clip_max)


def get_optimizer(type, model, **opt_kwargs):
    if hasattr(model.criterion, "parameters"):
        # Case: loss has trainable parameters; optimize these too.
        params_to_optimize = list(model.parameters()) + list(
            model.criterion.parameters())
    else:
        # Case: no trainable parameters in loss; just optimize model params.
        params_to_optimize = model.parameters()
    logging.info(f"fetching optimizer {type} with opt_kwargs {opt_kwargs}")
    if type == SGD_OPT:
        return torch.optim.SGD(params_to_optimize, **opt_kwargs)
    elif type == ADAM_OPT:
        return torch.optim.Adam(params_to_optimize, **opt_kwargs)
    elif type == ADAMW_OPT:
        return torch.optim.AdamW(params_to_optimize, **opt_kwargs)
    elif type == QHADAM_OPT:
        return QHAdam(params_to_optimize, **opt_kwargs)
    else:
        raise NotImplementedError
