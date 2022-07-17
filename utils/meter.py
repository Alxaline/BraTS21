# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""

import logging
from typing import Union, Sequence

import numpy as np

from utils.misc import apply_f

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name: str, fmt: str = "4f") -> None:
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.count = 0
        self.all_val = []

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.count = 0
        self.all_val = []

    def update(self, val: Union[float, int, np.ndarray], n: int = 1) -> None:
        self.val = val
        self.all_val.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.std(self.all_val, axis=0)

    def __sub__(self, other: 'AverageMeter') -> 'AverageMeter':
        assert self.name == other.name, f"For subtraction AverageMeter need to be the same"
        updated_meter = AverageMeter(self.name)
        updated_meter.avg = self.avg - other.avg
        updated_meter.std = self.std - other.std
        updated_meter.count = self.count - other.count
        return updated_meter

    def __str__(self) -> str:
        if isinstance(self.val, np.ndarray) and isinstance(self.avg, np.ndarray):
            fmtstr = f"{self.name} {apply_f(self.val.tolist(), lambda t: float(f'{t:{self.fmt}}'))}" \
                     f" ({apply_f(self.avg.tolist(), lambda t: float(f'{t:{self.fmt}}'))})"
        else:
            fmtstr = f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"
        return fmtstr


class ProgressMeter(object):
    """
    Progress Meter
    """

    def __init__(self, num_batches: int, meters: Sequence[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
