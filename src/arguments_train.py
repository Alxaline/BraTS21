# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import argparse
import os
from copy import deepcopy
from typing import Tuple, Dict

import oyaml as yaml

from utils.files import check_exist, check_isdir


def add_model_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Model configuration args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('model', 'Model configuration')
    group.add_argument("--model", default="equiunet",
                       choices=(
                           "basic_unet",
                           "nnunet",
                           "unet_tr",
                           "segresnet",
                           "segresnetvae",
                           "highresnet",
                           "vnet",
                           "modified_unet",
                           "att_unet",
                           "r2unet",
                           "r2attunet",
                           "equiunet",
                           "att_equiunet",
                           "equiunet_ref",
                           "equiunet_assp_evo",
                           "equiunet_assp_evo_ref",
                           "equiunet_assp_evocor",
                       # same as "equiunet_assp_evo", as be renamed after but needed to docker
                       ),
                       help="model to use (basic_unet | nnunet | unet_tr | segresnet | segresnetvae | highresnet | "
                            "vnet | modified_unet | att_unet | r2unet | r2attunet | equiunet | att_equiunet | "
                            "equiunet_ref | equiunet_assp_evo | equiunet_assp_evo_ref)")
    group.add_argument("--norm", type=str, default="instance", choices=["batch", "group", "instance", "bcn"])
    group.add_argument("--act", type=str, default="relu",
                       choices=["elu", "relu", "leakyrelu", "prelu", "swish", "mish"])
    group.add_argument("--width", type=int, default=48)
    group.add_argument("--dropout", type=float, default=0.0)
    group.add_argument("--num_classes", type=int, default=3)
    return parser


def add_data_loading_and_save_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Data loading args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('data', 'Data loading & save')
    group.add_argument("--train_data_path", type=check_isdir, required=True, help="path to the training data")
    group.add_argument("--val_data_path", type=check_isdir, default=None, help="path to the val data")
    group.add_argument("--already_preprocess", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default="./runs")
    parser.add_argument("--resume", type=check_exist, help='model .pth to restart from')
    parser.add_argument("--no_full_name", action="store_true", default=False)
    return parser


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Training hyper parameters args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('training', 'Training hyper parameters')
    group.add_argument("--batch_size", type=int, default=1)
    group.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128],
                       help="Patch size: default (128,128,128). "
                            "If (0,0,0) full image.")
    group.add_argument("--epochs", type=int, default=350)
    group.add_argument("--val_frequency", type=int, default=3)
    group.add_argument("--no_amp", action="store_true", default=False)
    group.add_argument("--criterion", type=str, default="dice",
                       choices=["dice", "generalized_dice", "focal", "tversky", "jaccard", "hd", "dice_hd", "boundary",
                                "dice_boundary", "dice_ce", "dice_ssim", "dice_focal"])
    group.add_argument("--gradient_accumulation_iter", type=int, default=None,
                       help="Iteration for gradient accumulation (Multiplicative factor of the actual BS)")
    group.add_argument("--adaptive_gradient_clipping", action="store_true", default=False)
    group.add_argument("--gradient_clipping", action="store_true", default=False)
    group.add_argument("--max_grad_norm", type=int, default=1,
                       help="Maximum gradient norm for gradient clipping (default: 1)")
    # group.add_argument("--add_min_max", action="store_true", default=False,
    #                    help="Z-score standardization + min max scaling with clipping")
    group.add_argument("--remove_outliers", action="store_true", default=False,
                       help="Remove outliers in zscore (clip std > 3)")
    group.add_argument("--num_workers", type=int, default=4)
    group.add_argument("--seed", type=int, default=123, help="random seed")
    group.add_argument("--fold", default=0, type=int, choices=[0, 1, 2, 3, 4, None],
                       help="Split number (0 to 4). If None is specified will use the full training set")
    group.add_argument("--device", type=str, default='0', help="device id for GPU")
    return parser


def add_optimizer_and_lr_scheduler_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Optimizer & learning rate scheduler args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('optimizer & lr', 'Optimizer & learning rate scheduler parameters')
    group.add_argument("--learning_rate", type=float, default=0.0001)
    group.add_argument("--weight_decay", type=float, default=0.00001)
    group.add_argument("--optimizer", default="ranger",
                       choices=("sgd", "adam", "adamw", "ranger", "ranger21", "novograd"),
                       help="optimizer to use (sgd | adam | adamw | ranger | ranger21 | novograd)")
    group.add_argument("--decay_type", default="flat_cosine",
                       choices=("step", "step_warmup", "cosine_warmup", "cosine", "flat_cosine"),
                       help="optimizer to use (step | step_warmup | cosine_warmup | flat_cosine")
    group.add_argument("--swa_start", type=int, default=None,
                       help="start stochastic weight averaging. Epoch were to start SWA.")
    group.add_argument("--swa_lr", type=float, default=0.00005)
    group.add_argument("--swa_anneal_epochs", type=int, default=10)
    return parser


def add_special_ranger_opt_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Special ranger Optimizer args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('Ranger options', 'Special Ranger optimizer options')
    group.add_argument("--use_gc", action="store_true", default=False)
    group.add_argument("--use_gcnorm", action="store_true", default=False)
    group.add_argument("--normloss", action="store_true", default=False)
    group.add_argument("--normloss_factor", type=float, default=1e-4)
    group.add_argument("--gc_conv_only", action="store_true", default=False)
    return parser


def add_log_and_metrics_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Log and metrics args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('log', 'Log and metrics parameters')
    group.add_argument("--log_train_interval", type=int, default=10)
    group.add_argument("--log_train_metrics", action="store_true", default=False)
    group.add_argument("--log_val_interval", type=int, default=1)
    group.add_argument("--log_val_metrics", action="store_true", default=False)
    group.add_argument("--key_metric", nargs=1, choices=["dice", "hausdorff_distance95"], default=["dice"])
    group.add_argument("--additional_metrics", nargs="+",
                       choices=["dice", "hausdorff_distance95", "sensitivity", "specificity", None],
                       default=["hausdorff_distance95"])
    group.add_argument("--save_on", type=str, default="loss", choices=["key_metric", "loss"])
    group.add_argument("--no_tensorboard", action="store_true", default=False)
    group.add_argument("--evaluate_end_training", action="store_true", default=False,
                       help="Evaluate model when training is done")
    group.add_argument("--only_evaluate", action="store_true", default=False,
                       help="Only evaluate the model")
    group.add_argument("--debug_val", action="store_true", default=False,
                       help="skip train and run val")
    group.add_argument("--sliding_window_inference", action="store_true", default=False,
                       help="Do sliding window for inference in validation")
    group.add_argument("--sliding_window_size", type=int, nargs=3, default=[128, 128, 128],
                       help="Roi size of the sliding window")
    group.add_argument("-v", "--verbosity", action="count", default=0,
                       help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def get_args() -> Tuple[argparse.Namespace, Dict[str, argparse.Namespace]]:
    """
    Parse all the args

    :return: args, arg_groups
    """
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Model Training')

    parser = add_model_config_args(parser)
    parser = add_data_loading_and_save_args(parser)
    parser = add_training_args(parser)
    parser = add_optimizer_and_lr_scheduler_args(parser)
    parser = add_special_ranger_opt_args(parser)
    parser = add_log_and_metrics_args(parser)

    args = parser.parse_args()

    if args.only_evaluate:
        assert args.resume, "if only_evaluate: the config file .yaml corresponding to resume args is needed"

    if args.gradient_accumulation_iter is not None:
        assert args.gradient_accumulation_iter > 0, "Number of iteration for gradient accumulation need to be > 0"

    if args.resume:
        resume_path = deepcopy(args.resume)
        save_path = os.path.join(os.path.dirname(args.resume))
        train_data_path = deepcopy(args.train_data_path)
        val_data_path = deepcopy(args.val_data_path)
        with open(os.path.join(os.path.dirname(args.resume), "config.yaml"), "r") as infile:
            loaded_config_file = yaml.safe_load(infile)
        args = argparse.Namespace(**loaded_config_file)
        args.resume = resume_path
        args.save_path = save_path
        args.train_data_path = train_data_path
        args.val_data_path = val_data_path

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    return args, arg_groups
