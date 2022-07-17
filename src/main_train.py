# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import gc
import logging
import os
import resource
import time
import warnings
from datetime import datetime

import numpy as np
import oyaml as yaml
import torch
from torch.optim.swa_utils import SWALR
from torch.utils.tensorboard import SummaryWriter

from learning.engine import Engine
from learning.lr_scheduler import AGC
from src import set_main_logger
from src.arguments_train import get_args
from src.definer import make_train_val_dataloader, make_experience_name, get_model, make_criterion, make_optimizer, \
    make_scheduler, get_post_transforms, get_activation, get_save_seg_transforms
from utils.meter import AverageMeter, ProgressMeter
from utils.visualization import metric_to_df

# avoid this issues: https://github.com/Project-MONAI/MONAI/issues/701
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def main() -> None:
    """
    main function for training
    """
    total_start = time.time()

    args, args_groups = get_args()

    # verify cuda is available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    # define save_path
    experience_name, h_params = make_experience_name(args_groups)
    original_save_path = args.save_path
    args.save_path = os.path.join(args.save_path,
                                  experience_name if not args.no_full_name else "") if not args.resume else os.path.dirname(
        args.resume)

    # create save_path
    try:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    except OSError:
        warnings.warn("Automatic file name with args is too long, replace with the current date and time")
        args.save_path = os.path.join(original_save_path, datetime.now().strftime(
            "%b%d_%H-%M-%S-%f")) if not args.resume else os.path.dirname(
            args.resume)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)

    # output config file
    if not args.resume:
        with open(os.path.join(args.save_path, "config.yaml"), "w") as outfile:
            yaml.dump(vars(args), outfile, default_flow_style=False)

    # set logger
    set_main_logger(log_file=True, filename=os.path.join(args.save_path, "logfile.log"), verbosity_lvl=args.verbosity)
    logger = logging.getLogger(__name__)

    # define dataloader (determinism is set inside function)
    train_loader, valid_loader = make_train_val_dataloader(args)
    # add arguments for ranger21
    args.num_batches_per_epoch = len(train_loader)

    # define model architecture
    model = get_model(args)

    # define loss criterion
    criterion = make_criterion(args)

    # define optimizer
    optimizer = make_optimizer(args, model)

    if args.adaptive_gradient_clipping:
        optimizer = AGC(model.parameters(), optimizer)

    # learning rate finder
    # import matplotlib.pyplot as plt
    # from monai.optimizers import LearningRateFinder
    # lower_lr, upper_lr = 1e-6, 1e-0
    # lr_finder = LearningRateFinder(model, optimizer, criterion)
    # lr_finder.range_test(train_loader, valid_loader, start_lr=lower_lr, end_lr=upper_lr, num_iter=20,
    #                      image_extractor=lambda x: x["img"],
    #                      label_extractor=lambda x: x["seg"])
    # steepest_lr, _ = lr_finder.get_steepest_gradient()
    # print(steepest_lr)
    # ax = plt.subplots(1, 1, figsize=(15, 15), facecolor="white")[1]
    # _ = lr_finder.plot(ax=ax)
    # plt.show()

    # define learning rate scheduler
    scheduler = make_scheduler(args, optimizer)

    # define loss scaler for automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # define model and learning rate scheduler for stochastic weight averaging
    swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa_start else None
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr, anneal_epochs=args.swa_anneal_epochs,
                          anneal_strategy='cos') if args.swa_start else None

    # define tensorboard writer
    summary_writer = SummaryWriter(args.save_path) if not args.no_tensorboard else None

    # define engine
    engine = Engine(model, criterion, args.num_classes, swa_model, key_metric=args.key_metric,
                    additional_metrics=args.additional_metrics, summary_writer=summary_writer)

    # resume from a checkpoint path
    start_epoch = 1
    best_value = None  # use to save checkpoint
    if args.resume is not None:
        start_epoch, optimizer, best_value = engine.resume(args, optimizer, filepath=args.resume)

    # define epoch progress meter
    epoch_time = AverageMeter("Time", "6.3f")
    progress = ProgressMeter(args.epochs, [epoch_time], prefix="Epoch: ")

    if not args.only_evaluate:
        # define training loop
        current_time = time.time()
        logger.info(f"Time elapsed before training: {current_time - total_start}")
        for epoch in range(start_epoch, args.epochs + 1):

            gc.collect()
            torch.cuda.empty_cache()

            if not args.debug_val:
                losses_train, batch_time_train, data_time_train, metric_time_train, \
                key_metric_dict_meter_train, additional_metrics_dict_meter_train = \
                    engine.train(train_loader,
                                 optimizer,
                                 scaler,
                                 epoch,
                                 args,
                                 get_post_transforms(args),
                                 get_activation(),
                                 scheduler,
                                 swa_scheduler)

            # display total train time
            epoch_time.update(time.time() - current_time)
            current_time = time.time()
            progress.display(epoch)

            if args.fold is not None and (epoch % args.val_frequency == 0):
                losses_val, batch_time_val, data_time_val, metric_time_val, \
                key_metric_dict_meter_val, additional_metrics_dict_meter_val = \
                    engine.evaluate(valid_loader,
                                    epoch,
                                    args,
                                    False,
                                    False,
                                    get_post_transforms(args),
                                    get_activation())

                # save model based on save_on, key_metric or loss
                best_value = engine.save_checkpoint(args,
                                                    epoch,
                                                    best_value,
                                                    losses_val,
                                                    key_metric_dict_meter_val,
                                                    model,
                                                    optimizer,
                                                    swa_model)

                if not args.no_tensorboard:
                    losses_overfit = losses_val - losses_train
                    summary_writer.add_scalar(f"Loss/overfit", losses_overfit.avg, epoch)

            # if full dataset training
            if args.fold is None:
                best_value = engine.save_checkpoint(args,
                                                    epoch,
                                                    best_value,
                                                    losses_train,
                                                    key_metric_dict_meter_train,
                                                    model,
                                                    optimizer,
                                                    swa_model)

    if args.evaluate_end_training or args.only_evaluate:
        # only execute swa if swa_start
        model = get_model(args)
        model = torch.optim.swa_utils.AveragedModel(model) if args.swa_start else model
        model_type = "swa_model" if args.swa_start else "model"
        model_name_to_load = "last_model.pth" if args.swa_start else "best_model.pth"
        model.load_state_dict(torch.load(os.path.join(args.save_path, model_name_to_load))[model_type])
        args.log_val_interval = 1
        log_tensorboard_hparam = not args.no_tensorboard
        args.no_tensorboard = True
        engine = Engine(model, criterion, args.num_classes, model, key_metric=args.key_metric,
                        additional_metrics=args.additional_metrics, summary_writer=summary_writer)
        # evaluate with and without TTA
        for idx, eval_type in enumerate(["", "_tta"]):
            logger.info(f"Starting Evaluation {eval_type}")
            if idx > 0:
                engine.val_step = 0
            losses_eval, _, _, _, key_metric_meter_eval, additional_metrics_meter_eval = \
                engine.evaluate(valid_loader,
                                0, args,
                                True if "tta" in eval_type else False,
                                True if args.swa_start else False,
                                get_post_transforms(args),
                                get_activation(),
                                True,
                                get_save_seg_transforms(),
                                os.path.join(args.save_path, f"segmentations{eval_type}"),
                                None,
                                f"Evaluation_swa{eval_type}" if args.swa_start else f"Evaluation{eval_type}",
                                "metric_val",
                                True,
                                True)
            if log_tensorboard_hparam:
                unpack_dict = {k: v for d in [losses_eval, key_metric_meter_eval, additional_metrics_meter_eval]
                               for k, v in (d.items() if hasattr(d, "items") else {d.name: d}.items() if d else ())}
                metric_dict = metric_to_df(list(unpack_dict.values()), engine.labels, np.nan,
                                           get_std=True).dropna(axis="columns").to_dict()
                metric_dict = {k: v[0] for k, v in metric_dict.items()}
                hparam_dict = {k: str(v) if not isinstance(v, (int, float, str, bool, torch.Tensor)) else v for k, v in
                               h_params.items()}

                # add h_params
                hparam_dict.update({"tta": True}) if eval_type == "tta" else hparam_dict.update({"tta": False})
                hparam_dict.update({"swa": True}) if args.swa_start else hparam_dict.update({"swa": False})

                summary_writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict,
                                           run_name=f"Evaluation{eval_type}")


if __name__ == "__main__":
    main()
