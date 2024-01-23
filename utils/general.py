import math
import torch
import numpy as np

from copy import deepcopy


def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel))


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


# class ModelEMA:
#     """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
#     Keep a moving average of everything in the model state_dict (parameters and buffers).
#     This is intended to allow functionality like
#     https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
#     A smoothed version of the weights is necessary for some training schemes to perform well.
#     This class is sensitive where it is initialized in the sequence of model init,
#     GPU assignment and distributed training wrappers.
#     """
#
#     def __init__(self, model, decay=0.9999, updates=0):
#         # Create EMA
#         self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
#         # if next(model.parameters()).device.type != 'cpu':
#         #     self.ema.half()  # FP16 EMA
#         self.updates = updates  # number of EMA updates
#         self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
#         for p in self.ema.parameters():
#             p.requires_grad_(False)
#
#     def update(self, model, burning=True):
#         # Update EMA parameters
#         with torch.no_grad():
#             self.updates += 1
#             if not burning:
#                 # self.updates += 1
#                 d = self.decay(self.updates)
#             else:
#                 d = 0.0
#             msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
#             for k, v in self.ema.state_dict().items():
#                 if v.dtype.is_floating_point:
#                     v *= d
#                     v += (1. - d) * msd[k].detach()
#
#     def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
#         # Update EMA attributes
#         copy_attr(self.ema, model, include, exclude)
#
#     def reset_updates(self, num=None):
#         if num is None:
#             num = 1
#         self.updates = num
def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model, decay=None):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            if decay is None:
                d = self.decay(self.updates)
            else:
                d = decay

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)
