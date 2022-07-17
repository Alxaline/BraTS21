# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
from collections import Sequence
from typing import Union, Callable, List

import numpy as np
import torch


def apply_f(_sequence: Sequence, f: Callable) -> Union[List, Sequence]:
    """
    Apply function to arbitrary Sequence (ie: nested list)

    :param _sequence: list to apply function
    :param f: function
    :return: list with applied function
    """
    if isinstance(_sequence, (list, tuple)):
        return list(map(lambda t: apply_f(t, f), _sequence))
    else:
        return f(_sequence)


def flatten(_list: Sequence) -> List:
    """
    flatten irregular list of list

    :param _list: irregular list
    :return: flatten list
    """
    result = []
    if isinstance(_list, (list, tuple)):
        for x in _list:
            result.extend(flatten(x))
    else:
        result.append(_list)
    return result


def tensor_to_array(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert tensor to array

    :param tensor: a torch tensor
    :return: a np array
    """
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor
