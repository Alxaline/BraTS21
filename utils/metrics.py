# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
from collections import OrderedDict
from functools import partial
from typing import Dict, Callable, Sequence, Union, Tuple, Optional

import numpy as np
import torch
from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric, \
    compute_roc_auc
from monai.metrics.confusion_matrix import compute_confusion_matrix_metric, check_confusion_matrix_metric_name
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction

from utils.misc import tensor_to_array, apply_f


def set_labels(labels: Union[Sequence[int], Dict[str, int], int]) -> Dict[str, int]:
    if isinstance(labels, int):
        labels = [labels]

    if isinstance(labels, (list, tuple)):
        labels = OrderedDict({str(k): k for k in labels})
    elif isinstance(labels, dict):
        labels = OrderedDict({str(k): v for k, v in labels.items()})

    # ordered by value
    labels = OrderedDict({k: v for k, v in sorted(labels.items(), key=lambda item: item[1])})
    return labels


def compute_metric_tensor(y_pred: torch.Tensor,
                          y: torch.Tensor,
                          callable_metric_dict: Dict[Callable, Sequence[str]],
                          y_probs: Optional[torch.tensor] = None
                          ) -> Tuple[OrderedDict, np.ndarray]:
    # MUST BE Batch SIZE = 1

    metrics_result = OrderedDict()
    confusion_matrix = None

    for callable_metric, metric in callable_metric_dict.items():

        n_len = len(y_pred.shape)
        reduce_axis = list(range(2, n_len))
        existence_y_pred_label = torch.amax(y_pred, dim=reduce_axis).type(torch.bool)
        existence_y_label = torch.amax(y, dim=reduce_axis).type(torch.bool)

        condition_to_worst = torch.logical_or(~existence_y_pred_label * existence_y_label,
                                              existence_y_pred_label * ~existence_y_label)
        condition_to_best = ~existence_y_pred_label * ~existence_y_label

        if type(callable_metric) in [DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric]:
            with np.errstate(invalid='ignore'):  # avoid FloatingPointError in multiply for compute hausdorff
                metric_value = callable_metric(y_pred, y).float()
            if type(callable_metric) == DiceMetric:
                # check if the label is empty for y_pred and y ! if it's the case: dice = 1
                value_to_replace_best = torch.tensor(1.0, device=metric_value.device)
                metric_value = torch.where(condition_to_best.to(metric_value.device), value_to_replace_best,
                                           metric_value)
                # check if existence of y_pred and not existence in y ! if it's the case: Dice = 1
                value_to_replace_worst = torch.tensor(0.0, device=metric_value.device)
                metric_value = torch.where(condition_to_worst.to(metric_value.device), value_to_replace_worst,
                                           metric_value)

            elif type(callable_metric) == HausdorffDistanceMetric:
                value_to_replace_best = torch.tensor(0.0, device=metric_value.device, dtype=torch.float)
                metric_value = torch.where(condition_to_best.to(metric_value.device), value_to_replace_best,
                                           metric_value)

                # check if existence of y_pred and not existence in y ! if it's the case: Hausdorff distance = max
                # distance in image
                p1 = torch.full((1, len(y.shape[2:])), 0.0, device=metric_value.device)
                p2 = torch.tensor((240, 240, 155), device=metric_value.device)  # BraTS (240, 240, 155) y.shape[2:]
                squared_dist = torch.sum((p1 - p2) ** 2, dim=1)
                value_to_replace = torch.sqrt(squared_dist).float()
                metric_value = torch.where(condition_to_worst.to(metric_value.device), value_to_replace, metric_value)

            elif type(callable_metric) == SurfaceDistanceMetric:
                raise NotImplemented("Not tested and not implemented for the moment")

            metrics_result[metric[0]] = tensor_to_array(metric_value)

        elif type(callable_metric) == ConfusionMatrixMetric:

            confusion_matrix = callable_metric(y_pred, y)
            for metric_name in metric:
                sub_confusion_matrix = compute_confusion_matrix_metric(metric_name.lower(),
                                                                       confusion_matrix)  # shape B,N

                # for only specificity and sensitivity
                metric_value, not_nans = do_metric_reduction(sub_confusion_matrix, MetricReduction.NONE)

                value_to_replace_best = torch.tensor(1.0, device=metric_value.device)
                metric_value = torch.where(condition_to_best.to(metric_value.device), value_to_replace_best,
                                           metric_value)
                value_to_replace_worst = torch.tensor(0.0, device=metric_value.device)
                metric_value = torch.where(condition_to_worst.to(metric_value.device), value_to_replace_worst,
                                           metric_value)

                metrics_result[metric_name] = tensor_to_array(metric_value)

            # binary, Need to change for multiclass
            tp = confusion_matrix[..., 0].squeeze().cpu().numpy()
            fp = confusion_matrix[..., 1].squeeze().cpu().numpy()
            tn = confusion_matrix[..., 2].squeeze().cpu().numpy()
            fn = confusion_matrix[..., 3].squeeze().cpu().numpy()
            confusion_matrix = np.array([[tp, fp],
                                         [fn, tn]])

        elif type(callable_metric) == partial:  # means compute_roc_auc ! warning need y as probs
            assert y_probs is not None, "y_probs is required for computing roc auc"
            assert y.ndim == 5 or y.ndim == 3, "must be BCWDH for segmentation or BCS for classification "
            assert y_probs.ndim == y.ndim, "y_probs and y_preds need to have same number of dim"
            metric_value = []
            for i in range(y.shape[1]):
                if condition_to_best[0][i]:
                    metric_val_roc = 1
                elif condition_to_worst[0][i]:
                    metric_val_roc = 0
                else:
                    metric_val_roc = callable_metric(torch.flatten(y_probs[0][i]),
                                                     torch.flatten(y[0][i]))  # to be (batch_size, )
                if not isinstance(metric_val_roc, np.ndarray):
                    metric_val_roc = np.array([metric_val_roc])
                metric_value.append(metric_val_roc)
            metric_value = np.hstack(metric_value)[None]

            metrics_result[metric[0]] = metric_value

    return metrics_result, confusion_matrix


def get_metric_callable(metrics_type: Sequence[str],
                        include_background: bool = True,
                        reduction: Union[MetricReduction, str] = MetricReduction.NONE
                        ) -> Dict[Callable, Sequence[str]]:
    confusion_matrix_metrics = list(get_confusion_matrix_metric(metrics_type).keys())
    other_metrics = list(set(metrics_type) - set(confusion_matrix_metrics))

    dict_callable_metric = OrderedDict()
    kwargs = {"include_background": include_background,
              "reduction": reduction}
    for metric in other_metrics:
        if metric.lower() == "hausdorff_distance95":
            metric_callable = HausdorffDistanceMetric(distance_metric="euclidean",
                                                      percentile=95,
                                                      **kwargs)
        elif metric.lower() == "surface_distance":
            metric_callable = SurfaceDistanceMetric(**kwargs)
        elif metric.lower() == "dice":
            metric_callable = DiceMetric(**kwargs)
        elif metric.lower() == "roc_auc":
            metric_callable = partial(compute_roc_auc, average="macro")
        else:
            raise NotImplementedError(f"the metric {metric} is not implemented.")

        dict_callable_metric[metric_callable] = [metric.title()]

    if confusion_matrix_metrics:
        metric_callable = ConfusionMatrixMetric(metric_name=confusion_matrix_metrics,
                                                compute_sample=False,
                                                **kwargs)
        dict_callable_metric[metric_callable] = apply_f(confusion_matrix_metrics, lambda x: x.title())
    return dict_callable_metric


def get_confusion_matrix_metric(metrics_type: Sequence[str]) -> Dict[str, str]:
    confusion_matrix_metric = {}
    for metric in metrics_type:
        try:
            metric_name = check_confusion_matrix_metric_name(metric_name=metric)
            confusion_matrix_metric[metric] = metric_name
        except NotImplementedError:
            pass

    return confusion_matrix_metric
