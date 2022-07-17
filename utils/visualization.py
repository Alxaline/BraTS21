# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import os
from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utils.files import append_df_to_excel
from utils.meter import AverageMeter


def log_tensorboard(writer: SummaryWriter, meters: Sequence[AverageMeter], labels: Dict[str, int], global_step: int,
                    step_mode: str = "metric", tag: Optional[str] = "metric") -> None:
    """
    log on tensorboard from the learning engine
    :param writer: a tensorboard summary writer
    :param meters: a list of AverageMeter
    :param global_step: global step value to record
    :param step_mode: step mode (if "metric" inside the string, use the name of AverageMeter)
    :param labels: a dict identifier the labels
    :param tag: a unique data identifier (if "metric" inside the string, use the name of AverageMeter)
    """
    # if tag is none use meter as name
    _tag, _step_mode = "", ""
    for meter in meters:
        if "metric" in tag:
            _tag = meter.name
        else:
            _tag = tag
        if "metric" in step_mode:
            _step_mode = meter.name
        else:
            _step_mode = step_mode
        avg = meter.avg
        if isinstance(avg, np.ndarray):
            if avg.shape[0] != 1:  # apply metric reduction mean if BS != 1 on batch
                avg = np.mean(avg, axis=0)
            if avg.ndim > 1:
                avg = np.squeeze(avg, axis=0)
            for idx_metric, metric_value in enumerate(avg):
                name_label = list(labels.keys())[idx_metric]
                if _step_mode and idx_metric == 0:
                    _step_mode += "_"
                writer.add_scalar(f"{_tag}/{_step_mode}{name_label}", scalar_value=metric_value,
                                  global_step=global_step)
        else:
            writer.add_scalar(f"{_tag}/{_step_mode}", scalar_value=avg, global_step=global_step)


def metric_to_df(meters: Sequence[AverageMeter], labels: Dict[str, int], global_step: int, step_mode: str = "metric",
                 tag: str = "metric", get_std: bool = False, get_current_val: bool = False) -> pd.DataFrame:
    """
    Metric into DataFrame
    :param meters: a list of AverageMeter
    :param labels: a dict identifier the labels
    :param global_step: global step value to record
    :param step_mode: step mode (if "metric" inside the string, use the name of AverageMeter)
    :param tag: a unique data identifier (if "metric" inside the string, use the name of AverageMeter)
    :param get_std: get standard deviation
    :param get_current_val: get the current val instead of the average in the meter
    :return: DataFrame
    """

    if get_std and get_current_val:
        raise ValueError("get_std is not possible with get_current_val")

    _tag, _step_mode = "", ""
    metric_dict = {}
    for meter in meters:
        if "metric" in tag:
            _tag = meter.name
        else:
            _tag = tag
        if "metric" in step_mode:
            _step_mode = meter.name
        else:
            _step_mode = step_mode

        avg = meter.avg if not get_current_val else meter.val

        std = np.array([0] * len(list(labels.keys())))
        if get_std:
            std = meter.std

        if isinstance(avg, np.ndarray):  # tricky metric is always nd array
            if avg.shape[0] != 1:  # apply metric reduction mean if BS != 1 on batch
                avg = np.mean(avg, axis=0)
                if get_std:
                    std = np.mean(std, axis=0)
            if avg.ndim > 1:
                avg = np.squeeze(avg, axis=0)

            if get_std:
                if std.ndim > 1:
                    std = np.squeeze(std, axis=0)

            for idx_metric, (avg_value, std_value) in enumerate(zip(avg, std)):
                name_label = list(labels.keys())[idx_metric]
                if _step_mode and idx_metric == 0:
                    _step_mode += "_"
                if get_std:
                    metric_to_get = {f"{_step_mode}avg_{name_label}": avg_value,
                                     f"{_step_mode}std_{name_label}": std_value}
                else:

                    metric_to_get = {f"{_step_mode}{name_label}": avg_value}
                metric_dict.update({"id": global_step, **metric_to_get})
        else:
            continue

    df = pd.DataFrame([metric_dict])
    return df


def log_xlsx_file(filepath: str, meters: Sequence[AverageMeter], labels: Dict[str, int], global_step: int,
                  step_mode: str = "metric", tag: Optional[str] = "metric") -> None:
    """
    log metrics into xlsx file
    :param filepath: a filepath were to save (filepath, tag + ".xlsx)
    :param meters: a list of AverageMeter
    :param labels: a dict identifier the labels
    :param global_step: global step value to record
    :param step_mode: step mode (if "metric" inside the string, use the name of AverageMeter)
    :param tag: a unique data identifier (if "metric" inside the string, use the name of AverageMeter)
    """
    df = metric_to_df(meters, labels, global_step, step_mode, tag="metric", get_current_val=True)
    filename = os.path.join(os.path.abspath(filepath), tag + ".xlsx")
    header = True if not os.path.exists(filename) else False
    append_df_to_excel(filename=filename, df=df, sheet_name="result", header=header, index=False)
    df_stat = pd.read_excel(io=filename, sheet_name="result", engine="openpyxl")
    append_df_to_excel(filename=filename, df=df_stat.describe(), truncate_sheet=False, sheet_name="stat", start_row=0,
                       header=True)
