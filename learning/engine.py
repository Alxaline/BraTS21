# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from typing import Union, Sequence, Dict, List, Tuple, Optional

import monai.transforms
import numpy as np
import torch
import torch.optim.lr_scheduler
import torch.optim.swa_utils
from monai.networks.utils import eval_mode
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

import tta
from src.definer import get_tta_transforms
from utils.files import segmentation_save, save_checkpoint
from utils.inferers import sliding_window_inference
from utils.meter import AverageMeter, ProgressMeter
from utils.metrics import set_labels, get_metric_callable, compute_metric_tensor
from utils.misc import flatten, apply_f
from utils.transforms import perform_staple_on_brats_multi_channel
from utils.transforms import shape_to_divisible, shape_to_original, pad_back_to_shape_before_compose, \
    remove_background_voxels
from utils.visualization import log_tensorboard, log_xlsx_file

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self,
                 model: Union[Sequence[torch.nn.Module], torch.nn.Module],
                 criterion: _Loss,
                 num_classes: int,
                 swa_model: Optional[torch.nn.Module] = None,
                 key_metric: Optional[List[str]] = None,
                 additional_metrics: Optional[Sequence[str]] = None,
                 summary_writer: Optional = None,
                 labels: Optional[Union[Sequence[int], Dict[str, int], int]] = None
                 ) -> None:

        self.model = model.cuda() if isinstance(model, torch.nn.Module) else model
        self.criterion = criterion.cuda()
        self.swa_model = swa_model.cuda() if swa_model is not None else swa_model
        self.key_metric = get_metric_callable(key_metric, include_background=True) if key_metric else key_metric
        self.additional_metrics = get_metric_callable(additional_metrics, include_background=True) \
            if additional_metrics else additional_metrics
        self.summary_writer = summary_writer
        if labels:
            assert len(labels) == len(num_classes)
        else:
            labels = list(range(num_classes))
        self.labels = set_labels(labels)
        self.train_step = 0
        self.val_step = 0
        self.test_step = 0

    def train(self,
              data_loader: DataLoader,
              optimizer: Union[torch.optim.Optimizer],
              scaler: torch.cuda.amp.GradScaler,
              epoch: int,
              args: argparse.Namespace,
              post_trans: Optional[monai.transforms.Compose] = None,
              activation: Optional[monai.transforms.Compose] = None,
              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              swa_scheduler: Optional[torch.optim.swa_utils.SWALR] = None
              ) -> Tuple[AverageMeter, AverageMeter, AverageMeter, Union[AverageMeter, None],
                         Union[Dict[str, AverageMeter], None], Union[Dict[str, AverageMeter], None]]:

        assert isinstance(self.model, torch.nn.Module), 'Model for train need ne to a torch.nn.Module'

        # set log meters
        progress, losses, batch_time, data_time, metric_time, key_metric_meter, additional_metrics_meter = \
            self._set_meters(args, len(data_loader), f"Epoch Train: [{epoch}]", "train")

        # switch to train mode
        self.model.train()

        current_time = time.time()
        for batch_idx, batch_data in enumerate(data_loader):
            image = batch_data["img"].cuda()
            target = batch_data["seg"].cuda()

            # was added for boundary loss
            if "boundary" in args.criterion:
                target = [target, batch_data["distance_map"].cuda()]

            # count train step
            self.train_step += 1

            # measure data loading time
            data_time.update(time.time() - current_time)

            # compute output
            if args.gradient_accumulation_iter is None:
                self.model.zero_grad()
            outputs = self._compute_output(args, model=self.model, img=image, is_train=True)
            outputs, loss = self._compute_loss(args, outputs, target)
            outputs = activation(outputs) if activation else outputs
            outputs = post_trans(outputs) if post_trans else outputs

            # if args.gradient_accumulation_iter is not None:
            #     loss = loss / args.gradient_accumulation_iter

            # record loss
            losses.update(loss.item())

            # Backward step
            scaler.scale(loss).backward()

            if args.gradient_accumulation_iter is None:
                self._unscale_and_clip(args, optimizer, scaler)
                scaler.step(optimizer)
                scaler.update()
            else:
                if (batch_idx + 1) % args.gradient_accumulation_iter == 0:
                    self._unscale_and_clip(args, optimizer, scaler)
                    scaler.step(optimizer)
                    scaler.update()
                    self.model.zero_grad()
                else:
                    pass

            # measure elapsed time
            batch_time.update(time.time() - current_time)
            current_time = time.time()

            # compute and log metrics
            metric_time, key_metric_meter, additional_metrics_meter = \
                self._log_metrics(args, outputs, target if not isinstance(target, Sequence) else target[0], metric_time,
                                  key_metric_meter,
                                  additional_metrics_meter, "train")

            # display and log tensorboard - step (last step is display = equivalent to epoch)
            if ((batch_idx + 1) % args.log_train_interval == 0) or ((batch_idx + 1) == len(data_loader)):
                progress.display(batch_idx + 1)
                self._log_tensorboard(args, self.train_step, progress.meters, step_mode="train_step", tag="metric")

        # log last step = epoch in tensorboard
        self._log_tensorboard(args, epoch, progress.meters, step_mode="train_epoch", tag="metric")
        self.summary_writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)

        if scheduler is not None:
            if args.swa_start is not None and epoch <= args.swa_start:
                pass
            else:
                scheduler.step()

        if args.swa_start is not None and epoch > args.swa_start:
            self.swa_model.update_parameters(self.model)
            swa_scheduler.step()

        return losses, batch_time, data_time, metric_time, key_metric_meter, additional_metrics_meter

    def evaluate(self,
                 data_loader: DataLoader,
                 epoch: int,
                 args: argparse.Namespace,
                 use_tta: bool = False,
                 force_swa_model: bool = False,
                 post_trans: Optional[monai.transforms.Compose] = None,
                 activation: Optional[monai.transforms.Compose] = None,
                 save_seg: bool = False,
                 save_seg_trans: Optional[monai.transforms.Compose] = None,
                 output_path: Optional[str] = None,
                 suffix_seg_name: Optional[str] = None,
                 tag: str = "metric",
                 step_mode: str = "val",
                 force_xlsx_save: bool = False,
                 return_original_shape: bool = False
                 ) -> Tuple[AverageMeter, AverageMeter, AverageMeter,
                            Union[AverageMeter, None],
                            Union[Dict[str, AverageMeter], None],
                            Union[Dict[str, AverageMeter], None]]:

        if (self.swa_model is not None and args.swa_start is not None and epoch > args.swa_start) or force_swa_model:
            models = [self.swa_model]
            step_mode_suff = "_swa"
        else:
            models = [self.model] if not isinstance(self.model, (tuple, list)) else self.model
            step_mode_suff = ""
        step_mode = f"{step_mode}{step_mode_suff}"

        k_divisible = 8
        # can be 8 if no refinement module

        # set log meters
        progress, losses, batch_time, data_time, metric_time, key_metric_meter, additional_metrics_meter = \
            self._set_meters(args, len(data_loader), f"Epoch Valid: [{epoch}]", step_mode=step_mode)

        # switch to eval mode
        current_time = time.time()

        for batch_idx, batch_data in enumerate(data_loader):
            image = batch_data["img"].cuda()
            target = batch_data["seg"].cuda() if "seg" in batch_data else None

            # was added for boundary loss
            if "boundary" in args.criterion:
                target = [target, batch_data["distance_map"].cuda()]

            # count val step
            self.val_step += 1

            # measure data loading time
            data_time.update(time.time() - current_time)

            # reshape images to fit to model
            image, p_b, p_a = shape_to_divisible(image, k=k_divisible)
            if target is not None:
                target = apply_f(target, lambda x: shape_to_divisible(x, k=k_divisible)[0])
                # target, _, _ = shape_to_divisible(target, k=8)

            outputs_models, loss_models = [], []
            for model in models:
                model = model.cuda() if not next(model.parameters()).is_cuda else model

                with eval_mode(model):
                    # compute output
                    if use_tta:
                        outputs_tta = self._apply_tta(args, model=model, img=image, tta_transforms=get_tta_transforms())
                        outputs_models.extend(outputs_tta)

                    else:
                        outputs = self._compute_output(args, model=model, img=image)
                        outputs_models.append(apply_f(outputs, lambda x: x.cpu() if len(models) != 1 else x))

            with torch.no_grad():
                outputs_loss = list(map(lambda x: self._compute_loss(args, x, target), outputs_models))
                outputs, loss = zip(*outputs_loss)
                outputs, loss = apply_f(outputs, lambda x: activation(x.to(image.device)).cpu() if (
                        len(models) != 1 or use_tta) else activation(x)), torch.stack(loss).mean(
                    dim=0) if target is not None else None  # not the moment efficient to switch cpu to gpu but
                # needed due to memory limitation and running activation with Half on cpu is not implemented

            if hasattr(args, "perform_staple") and args.perform_staple:
                outputs = perform_staple_on_brats_multi_channel(datas=apply_f(outputs, post_trans),
                                                                threshold_value=args.staple_threshold,
                                                                return_as_tensor=True)
            else:  # Do mean of prediction
                outputs = post_trans(torch.stack(outputs).mean(dim=0))

            # back outputs and loss on image device
            outputs, loss = outputs.to(image.device), loss.to(image.device) if target is not None else None

            # record loss
            if target is not None:
                losses.update(loss.item())

            # remove potential voxels in background
            outputs = remove_background_voxels(img=image, outputs=outputs.to(image.device))

            # measure elapsed time
            batch_time.update(time.time() - current_time)
            current_time = time.time()

            # log metrics
            metric_time, key_metric_meter, additional_metrics_meter = \
                self._log_metrics(args, outputs, target if not isinstance(target, Sequence) else target[0],
                                  metric_time, key_metric_meter, additional_metrics_meter, step_mode)

            # display and log tensorboard - step (last step is display = equivalent to epoch)
            if ((batch_idx + 1) % args.log_val_interval == 0) or ((batch_idx + 1) == len(data_loader)):
                progress.display(batch_idx + 1)
                self._log_tensorboard(args, self.val_step, progress.meters, step_mode=f"{step_mode}_step",
                                      tag=tag,
                                      force_xlsx_save=force_xlsx_save, patient_id=batch_data["patient_id"][0])

            # log last step = epoch in tensorboard
            if (batch_idx + 1) == len(data_loader):
                self._log_tensorboard(args, epoch, progress.meters, step_mode=f"{step_mode}_epoch", tag="metric")

            if return_original_shape:
                outputs = shape_to_original(outputs, p_b, p_a)
                outputs = save_seg_trans(outputs) if save_seg_trans is not None else outputs
                outputs = pad_back_to_shape_before_compose(batch_data, outputs)

            # save outputs seg labels
            if save_seg:
                assert output_path, "if save_seg, output_path is needed"
                segmentation_save(batch_data,
                                  outputs,
                                  output_path if hasattr(args, "create_patient_dir") and not args.create_patient_dir
                                  else os.path.join(output_path,
                                                    batch_data["patient_id"][0]),
                                  suffix=suffix_seg_name)

        return losses, batch_time, data_time, metric_time, key_metric_meter, additional_metrics_meter

    @staticmethod
    def _compute_output(args: argparse.Namespace,
                        model: torch.nn.Module,
                        img: torch.Tensor,
                        is_train: bool = False) -> torch.Tensor:

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            if args.sliding_window_inference and not is_train:
                outputs = sliding_window_inference(img, roi_size=args.sliding_window_size, sw_batch_size=1,
                                                   predictor=model, device=torch.device("cpu"))
            else:
                outputs = model(img)
        return outputs

    def _compute_loss(self,
                      args: argparse.Namespace,
                      outputs: torch.Tensor,
                      label: Optional[torch.Tensor] = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:

        if label is not None:
            device = label[0].device if isinstance(label, Sequence) else label.device

        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            if isinstance(outputs, (tuple, list)):  # use of Deep Supervision
                outputs = flatten(outputs)
                deeps = outputs[1:]
                del outputs[1:]
                loss = torch.mean(
                    torch.stack(
                        [self.criterion(deep.to(device), label) for deep in
                         outputs + deeps])) if label is not None else None
                outputs = outputs[0]
            else:
                loss = self.criterion(outputs.to(device), label) if label is not None else None
        return outputs, loss

    def _set_meters(self,
                    args: argparse.Namespace,
                    total_length: int,
                    prefix_progress: str,
                    step_mode: str = "train"
                    ) -> Tuple[ProgressMeter, AverageMeter, AverageMeter, AverageMeter,
                               Union[AverageMeter, None], Union[Dict[str, AverageMeter], None],
                               Union[Dict[str, AverageMeter], None]]:

        assert any(x in step_mode for x in ["train", "val", "test"]), "step_mode need to be 'train', 'val' or 'test'"

        losses = AverageMeter(name="Loss", fmt="6.4f")
        batch_time = AverageMeter("Time", "6.3f")
        data_time = AverageMeter("Data", "6.3f")
        display = [batch_time, data_time, losses]
        progress = ProgressMeter(total_length, display, prefix=prefix_progress)

        metric_time, key_metric_meter, additional_metrics_meter = None, None, None
        if (args.log_train_metrics and "train" in step_mode) or (args.log_val_metrics and "val" in step_mode):
            key_metric_meter = OrderedDict(
                {m: AverageMeter(name=m, fmt="8.3f") for m in flatten(list(self.key_metric.values()))})
            metric_time = AverageMeter("Metric Time", "6.3f")
            display.insert(2, metric_time)
            display.extend(list(key_metric_meter.values()))

            if self.additional_metrics:
                additional_metrics_meter = OrderedDict({m: AverageMeter(name=m, fmt="8.3f") for m in
                                                        flatten(list(self.additional_metrics.values()))})
                display.extend(list(additional_metrics_meter.values()))

            progress = ProgressMeter(total_length, display, prefix=prefix_progress)

        return progress, losses, batch_time, data_time, metric_time, key_metric_meter, additional_metrics_meter

    def _compute_metrics(self,
                         y_pred: torch.Tensor,
                         y: torch.Tensor,
                         key_metric_meter: Union[Dict[str, AverageMeter], None],
                         additional_metrics_meter: Union[Dict[str, AverageMeter], None]
                         ) -> Tuple[Union[Dict[str, AverageMeter], None], Union[Dict[str, AverageMeter], None]]:

        if self.key_metric:
            key_metric_result, _ = compute_metric_tensor(y_pred, y, callable_metric_dict=self.key_metric)
            for m in key_metric_meter:
                key_metric_meter[m].update(key_metric_result[m])
        if self.additional_metrics:
            additional_metrics_result, _ = compute_metric_tensor(y_pred, y,
                                                                 callable_metric_dict=self.additional_metrics)
            for m in additional_metrics_meter:
                additional_metrics_meter[m].update(additional_metrics_result[m])

        return key_metric_meter, additional_metrics_meter

    def _log_metrics(self,
                     args: argparse.Namespace,
                     output: torch.Tensor,
                     target: torch.Tensor,
                     metric_time: AverageMeter,
                     key_metric_meter: Union[Dict[str, AverageMeter], None],
                     additional_metrics_meter: Union[Dict[str, AverageMeter], None],
                     step_mode: str = "train"
                     ) -> Tuple[AverageMeter, Union[Dict[str, AverageMeter], None],
                                Union[Dict[str, AverageMeter], None]]:

        if (args.log_train_metrics and "train" in step_mode) or (args.log_val_metrics and "val" in step_mode):
            current_time = time.time()
            key_metric_meter, additional_metrics_meter = self._compute_metrics(output, target, key_metric_meter,
                                                                               additional_metrics_meter)
            metric_time.update(time.time() - current_time)

        return metric_time, key_metric_meter, additional_metrics_meter

    def _log_tensorboard(self,
                         args: argparse.Namespace,
                         batch_idx: int, meters: Sequence[AverageMeter],
                         step_mode: str,
                         tag: str,
                         force_xlsx_save: Optional[bool] = False,
                         patient_id=Optional[str]
                         ) -> None:

        if not args.no_tensorboard:
            log_tensorboard(writer=self.summary_writer, meters=meters, labels=self.labels, global_step=batch_idx,
                            step_mode=step_mode, tag=tag)
        if force_xlsx_save:
            log_xlsx_file(args.save_path, meters=meters, labels=self.labels,
                          global_step=batch_idx if not patient_id else patient_id,
                          step_mode="metric", tag=tag)

    def _apply_tta(self,
                   args: argparse.Namespace,
                   model: torch.nn.Module,
                   img: torch.Tensor,
                   tta_transforms: Optional[tta.Compose]
                   ) -> List[Union[list, Sequence]]:

        outputs_tta, loss_tta = [], []
        for transformer_tta in tta_transforms:
            # augment image
            augmented_image = transformer_tta.augment_image(img)
            # compute output
            outputs = self._compute_output(args, model, augmented_image)
            # reverse augmentation for outputs
            deaug_outputs = apply_f(outputs, lambda x: transformer_tta.deaugment_mask(x).cpu())
            outputs_tta.append(deaug_outputs)  # due to memory limitation switch to .cpu()
        return outputs_tta

    def _unscale_and_clip(self, args: argparse.Namespace,
                          optimizer: Union[torch.optim.Optimizer],
                          scaler: torch.cuda.amp.GradScaler):

        if args.gradient_clipping or args.adaptive_gradient_clipping:
            if not args.no_amp:
                scaler.unscale_(optimizer)
            if args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               args.max_grad_norm)
            # in FP16 Gradients are scaled, so we can clip to max_norm*scale or unscale

    @staticmethod
    def save_checkpoint(args: argparse.Namespace,
                        epoch: int,
                        best_value: float,
                        loss_average_meter: AverageMeter,
                        key_metric_meter_dict: Union[Dict[str, AverageMeter], None],
                        model: torch.nn.Module,
                        optimizer: Union[torch.optim.Optimizer],
                        swa_model: Optional[torch.nn.Module] = None,
                        ) -> Union[int, float, np.ndarray]:

        if args.fold is None and not args.log_train_metrics and args.save_on == "key_metric":
            raise ValueError("Run a full training by specifying 'fold = None', is incompatible with "
                             "'save_on = key_metric' if you don't specified log_train_metrics")

        if args.fold is not None and not args.log_val_metrics and args.save_on == "key_metric":
            raise ValueError("log_val_metrics need to be specified if, is incompatible with 'save_on = key_metric'")

        # BS always 1 for val
        do_save_best_model = False
        if args.save_on == "loss" or (args.save_on == "key_metric" and args.key_metric == "hausdorff_distance95"):
            # hausdorfff_distance95 need to be minimize
            if best_value:
                if loss_average_meter.avg < best_value:
                    best_value = loss_average_meter.avg
                    do_save_best_model = True
                else:
                    do_save_best_model = False
            else:
                best_value = loss_average_meter.avg
                do_save_best_model = True
        elif args.save_on == "key_metric":
            key_metric_average_meter = key_metric_meter_dict[args.key_metric]
            if key_metric_average_meter.avg.size > 1:
                key_metric_average_meter = key_metric_average_meter.avg.sum(axis=1)
            if best_value:
                if key_metric_average_meter.avg > best_value:
                    best_value = key_metric_average_meter.avg
                    do_save_best_model = True
                else:
                    do_save_best_model = False
            else:
                best_value = key_metric_average_meter.avg
                do_save_best_model = True

        kwargs = {
            "model": model.state_dict(),
            "swa_model": swa_model.state_dict() if swa_model else None,
            "optimizer": optimizer.state_dict(),
            args.save_on: best_value}
        if do_save_best_model:
            save_checkpoint(os.path.join(args.save_path, "best_model.pth"), epoch, **kwargs)

        save_checkpoint(os.path.join(args.save_path, "last_model.pth"), epoch, **kwargs)

        return best_value

    def resume(self,
               args: argparse.Namespace,
               optimizer: Union[torch.optim.Optimizer],
               filepath: str
               ) -> Tuple[int, Union[torch.optim.Optimizer], Union[int, float, np.ndarray]]:

        logger.info(f"Resume training from {args.resume}")
        checkpoint = torch.load(filepath)
        start_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model"], strict=False)
        if checkpoint["swa_model"]:
            self.swa_model.load_state_dict(checkpoint["swa_model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_value = checkpoint[args.save_on]
        return start_epoch, optimizer, best_value
