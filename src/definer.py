# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import argparse
import re
from datetime import datetime
from typing import Dict, Tuple, Union, Optional, Any, List

import torch
from monai import transforms as tr
from monai.data import Dataset, list_data_collate
from monai.losses import DiceLoss, GeneralizedDiceLoss, FocalLoss, TverskyLoss, DiceFocalLoss
from monai.networks.nets import BasicUNet, SegResNet, SegResNetVAE, HighResNet, VNet, UNETR
from monai.optimizers import Novograd
from monai.utils.misc import set_determinism
from ranger21.ranger21 import Ranger21
from sklearn.model_selection import KFold
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

import tta
from learning.losses import HausdorffLoss, DiceHDLoss, BoundaryLoss, DiceBoundaryLoss, DiceCELoss
from learning.lr_scheduler import GradualWarmupScheduler, FlatplusAnneal
# from ranger.ranger2020 import Ranger
from learning.optimizer import Ranger2020
from networks.equiunet2020 import EquiUnet, AttEquiUnet
from networks.equiunet2021 import EquiUnetASSPEvo
from networks.unet_family import Unet, R2Unet, AttUnet, R2AttUnet, WrapperDynUNet
from utils import transforms as tr_supp
from utils.files import create_database, create_database_test_docker, safe_file_name


def get_model(args: argparse.Namespace) -> torch.nn.Module:
    """
    get model

    :param args: ``argparse``
    :return: a pytorch model
    """
    model = None
    dimensions = 3
    in_channels = 4
    num_classes = args.num_classes
    dropout = args.dropout
    if args.model == "basic_unet":
        model = BasicUNet
        features = [args.width * 2 ** i for i in range(5)]
        features.append(features[0])
        kwargs = {"dimensions": dimensions,
                  "in_channels": in_channels,
                  "out_channels": num_classes,
                  "features": features,
                  "act": args.act,
                  "norm": args.norm,
                  "dropout": dropout
                  }
    elif args.model in ["nnunet"]:
        model = WrapperDynUNet
        kwargs = {"spatial_dims": dimensions,
                  "in_channels": in_channels,
                  "out_channels": num_classes,
                  "norm_name": args.norm
                  }
    elif args.model == "unet_tr":
        if args.norm == "group":
            norm = ("GROUP", {"num_groups": 8})
        else:
            norm = args.norm
        model = UNETR
        kwargs = {"feature_size": 16,
                  "hidden_size": 768,
                  "mlp_dim": 3072,
                  "num_heads": 12,
                  "pos_embed": "perceptron",
                  "img_size": args.patch_size,
                  "in_channels": in_channels,
                  "out_channels": num_classes,
                  "norm_name": norm
                  }

    elif args.model in ["segresnet", "segresnetvae"]:
        if args.model == "segresnet":
            model = SegResNet
            if args.norm == "group":
                norm = ("GROUP", {"num_groups": 8})
            else:
                norm = args.norm
            kwargs = {
                "spatial_dims": dimensions,
                "init_filters": 8,
                "in_channels": in_channels,
                "out_channels": num_classes,
                "dropout_prob": dropout,
                "norm": norm,
            }
        if args.model == "segresnetvae":
            model = SegResNetVAE
            kwargs = {"input_image_size": args.patch_size,
                      "spatial_dims": dimensions,
                      "init_filters": 8,
                      "in_channels": in_channels,
                      "out_channels": num_classes,
                      "blocks_down": [1, 2, 2, 4],
                      "blocks_up": [1, 1, 1],
                      "dropout_prob": dropout
                      }
    elif args.model == "highresnet":
        model = HighResNet
        kwargs = {"spatial_dims": dimensions,
                  "in_channels": in_channels,
                  "out_channels": num_classes,
                  "acti_type": args.act,
                  "norm_type": args.norm,
                  "dropout_prob": dropout
                  }
    elif args.model == "vnet":
        model = VNet
        kwargs = {"spatial_dims": dimensions,
                  "in_channels": in_channels,
                  "out_channels": num_classes,
                  "act": args.act,
                  "dropout_prob": dropout
                  }
    elif args.model in ["equiunet",
                        "att_equiunet",
                        "equiunet_ref",
                        "equiunet_assp_evo",
                        "equiunet_assp_evocor",  # same as "equiunet_assp_evo", as be renamed after but needed to docker
                        "equiunet_assp_evo_ref",
                        ]:
        kwargs = {"inplanes": in_channels,
                  "num_classes": num_classes,
                  "features": [args.width * 2 ** i for i in range(4)],
                  "norm_layer": args.norm,
                  "act": args.act,
                  "deep_supervision": True,
                  "dropout": dropout,
                  }
        if args.model == "equiunet":
            model = EquiUnet
        elif args.model == "equiunet_ref":
            model = EquiUnet
            kwargs.update({"refinement": True})
        elif args.model == "att_equiunet":
            model = AttEquiUnet
        elif args.model in ["equiunet_assp_evo", "equiunet_assp_evocor"]:
            model = EquiUnetASSPEvo
        elif args.model == "equiunet_assp_evo_ref":
            model = EquiUnetASSPEvo
            kwargs.update({"refinement": True})
    elif args.model in ["modified_unet", "att_unet", "r2unet", "r2attunet"]:
        if args.model == "modified_unet":
            model = Unet
        elif args.model == "att_equiunet":
            model = AttUnet
        elif args.model == "r2unet":
            model = R2Unet
        elif args.model == "r2attunet":
            model = R2AttUnet
        kwargs = {"img_ch": in_channels,
                  "output_ch": num_classes,
                  "features": [args.width * 2 ** i for i in range(4)],
                  "norm_layer": args.norm,
                  "act": args.act,
                  "deep_supervision": True,
                  }

    else:
        raise NameError("Not Supported Model")
    return model(**kwargs)


def make_criterion(args: argparse.Namespace) -> _Loss:
    """
    get criterion

    :param args: argparse
    :return: a pytorch _Loss
    """
    if args.criterion == "dice":
        criterion_function = DiceLoss
        kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "squared_pred": True,
            "jaccard": False,
            "batch": True,
        }
    elif args.criterion == "jaccard":
        criterion_function = DiceLoss
        kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "squared_pred": True,
            "jaccard": True,
            "batch": True,
        }
    elif args.criterion == "dice_ce":
        criterion_function = DiceCELoss
        kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "squared_pred": True,
            "batch": True,
        }
    elif args.criterion == "dice_focal":
        criterion_function = DiceFocalLoss
        kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "squared_pred": True,
            "batch": False,
        }

    elif args.criterion == "generalized_dice":
        criterion_function = GeneralizedDiceLoss
        kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "squared_pred": True,
            "w_type": "square",
        }
    elif args.criterion == "focal":
        criterion_function = FocalLoss
        kwargs = {
            "gamma": 2.0,
        }
    elif args.criterion == "tversky":
        criterion_function = TverskyLoss
        kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "alpha": 0.5,
            "beta": 0.5,
        }
    elif args.criterion == "hd":
        criterion_function = HausdorffLoss
        kwargs = {
            "idc": list(range(args.num_classes)),
            "sigmoid": True,
            "softmax": False,
            "alpha": 2,
        }
    elif args.criterion == "dice_hd":
        criterion_function = DiceHDLoss
        kwargs = {
            "idc_hd": list(range(args.num_classes)),
            "alpha_hd": 2,
            "hybrid": False,
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "squared_pred": True,
            "weight_hd": 0.5,
            "weight_dice": 0.5,
        }
    elif args.criterion == "boundary":
        criterion_function = BoundaryLoss
        kwargs = {
            "idc": list(range(args.num_classes)),
            "sigmoid": True,
            "softmax": False,
        }
    elif args.criterion == "dice_boundary":
        criterion_function = DiceBoundaryLoss
        kwargs = {
            "idc_boundary": list(range(args.num_classes)),
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "squared_pred": True,
        }
    else:
        raise NameError("Not Supported Criterion")

    kwargs["reduction"] = "mean"

    return criterion_function(**kwargs)


def make_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> Union[torch.optim.Optimizer]:
    """
    get optimizer

    :param args: ``argparse``
    :param model: a pytorch model
    :return: a torch.optim.Optimizer
    """
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    kwargs = {}
    if args.optimizer == "sgd":
        optimizer_function = SGD
        kwargs = {"momentum": 0.9}
    elif args.optimizer == "adam":
        optimizer_function = Adam
        kwargs = {
            "betas": (0.9, 0.999),
            "eps": 1e-08
        }
    elif args.optimizer == "adamw":
        optimizer_function = AdamW
        kwargs = {
            "betas": (0.9, 0.999),
            "eps": 1e-08
        }
    elif args.optimizer == "ranger":
        optimizer_function = Ranger2020
        kwargs = {"alpha": 0.5,
                  "k": 6,
                  "N_sma_threshhold": 5,  # Ranger options
                  "betas": (.95, 0.999),
                  "eps": 1e-5,
                  "weight_decay": 0,  # Adam options
                  # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                  "use_gc": args.use_gc,
                  "use_gcnorm": args.use_gcnorm,
                  "normloss": args.normloss,
                  "normloss_factor": args.normloss_factor,
                  "gc_conv_only": args.gc_conv_only,
                  "gc_loc": True
                  }
    elif args.optimizer == "ranger21":
        optimizer_function = Ranger21
        kwargs = {
            "lookahead_active": True,
            "lookahead_mergetime": 5,
            "lookahead_blending_alpha": 0.5,
            "lookahead_load_at_validation": False,
            "use_madgrad": False,
            "use_adabelief": False,
            "using_gc": args.use_gc,
            "gc_conv_only": args.gc_conv_only,
            "normloss_active": args.normloss,
            "normloss_factor": args.normloss_factor,
            "use_adaptive_gradient_clipping": False,
            "agc_clipping_value": 1e-2,
            "agc_eps": 1e-3,
            "betas": (0.9, 0.999),  # temp for checking tuned warmups
            "momentum_type": "pnm",
            "pnm_momentum_factor": 1.0,
            "momentum": 0.9,
            "eps": 1e-8,
            "num_batches_per_epoch": args.num_batches_per_epoch,
            "num_epochs": args.epochs,
            "use_cheb": False,
            "use_warmup": False,
            "num_warmup_iterations": None,
            "warmdown_active": False,
            "warmdown_start_pct": 0.72,
            "warmdown_min_lr": 3e-5,
            "weight_decay": 1e-4,
            "decay_type": "stable",
            "warmup_type": "linear",
            "warmup_pct_default": 0.22,
            "logging_active": True,
        }

    elif args.optimizer == "novograd":
        optimizer_function = Novograd
        kwargs = {
            "betas": (0.9, 0.98),
            "eps": 1e-08
        }
    else:
        raise NameError("Not Supported Optimizer")

    kwargs["lr"] = args.learning_rate
    kwargs["weight_decay"] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args: argparse.Namespace,
                   optimizer: Union[torch.optim.Optimizer]) -> torch.optim.lr_scheduler._LRScheduler:
    """
    get scheduler

    :param args: ``argparse``
    :param optimizer: a ``torch.optim.Optimizer`` or ``Ranger`` or ``Novograd``
    :return: a torch.optim.lr_scheduler._LRScheduler
    """
    if args.decay_type == "step":
        scheduler = MultiStepLR(optimizer, milestones=list(range(30, args.epochs, 30)), gamma=0.1)
    elif args.decay_type == "step_warmup":
        scheduler = MultiStepLR(optimizer, milestones=list(range(30, args.epochs, 30)), gamma=0.1)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    elif args.decay_type == "cosine_warmup":
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.epochs // 20,
                                           after_scheduler=cosine_scheduler)
    elif args.decay_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    elif args.decay_type == "flat_cosine":
        scheduler = FlatplusAnneal(optimizer, max_iter=args.epochs, step_size=0.70, eta_min=0)
    else:
        raise Exception("unknown lr scheduler: {}".format(args.decay_type))

    return scheduler


def make_experience_name(args_group: Dict[str, argparse.Namespace], add_current_date_time: bool = True) \
        -> Tuple[str, Dict]:
    """
    get experience name

    :param args_group: Dict with category of ``argparse``
    :param add_current_date_time: add current date and time to experience name
    :return: experience name, dict of h_params
    """
    current_date_time = datetime.now().strftime("%b%d_%H-%M-%S-%f")
    category_in_name = ["model", "training", "optimizer & lr", "processing"]
    result = []
    h_params = {}
    for category in category_in_name:
        if category in args_group:
            h_params.update(sorted(vars(args_group[category]).items()))
            result.append("_".join(
                str(k)[0:3] + "_" + str(v)[0:3] if len(str(k).split("_")) == 1 else "".join(
                    [x[:1] for x in str(k).split("_")]) + "_" + str(v) for k, v in
                sorted(vars(args_group[category]).items())))
    experience_name = "_".join(result)
    experience_name = re.sub("_+", "_", safe_file_name(experience_name))
    if add_current_date_time:
        experience_name = current_date_time + "_" + experience_name
    return experience_name, h_params


def make_train_val_dataloader(args: argparse.Namespace, train_in_val_mode: bool = False) \
        -> Tuple[Union[DataLoader, None], Union[DataLoader, None]]:
    """
    get train val dataloader

    :param args: ``argparse``
    :param train_in_val_mode: make train data with transform of validation
    :return: train_loader, val_loader
    """

    train_transforms = tr.Compose([
        tr.LoadImaged(keys=["img", "seg"], reader="nibabelreader"),
        tr.ConvertToMultiChannelBasedOnBratsClassesd(keys=["seg"]),
        tr.CropForegroundd(keys=["img", "seg"], source_key="img"),
        tr.SpatialPadd(keys=["img", "seg"], spatial_size=args.patch_size),

        # add for the distance map transform (usefull for boundary loss)
        tr.CopyItemsd(keys=["seg"], times=1, names=["distance_map"]),
        tr_supp.OneHotToDistd(keys=["distance_map"], sampling=[1, 1, 1]),

        tr.RandSpatialCropd(keys=["img", "seg"], roi_size=args.patch_size, random_size=False),
        tr.RandRotate90d(keys=["img", "seg"], prob=0.7, spatial_axes=(0, 2)),
        tr.RandFlipd(keys=["img", "seg"], prob=0.7, spatial_axis=(0, 1, 2)),
        tr.RandShiftIntensityd(keys=["img"], prob=0.7, offsets=0.1),
        tr.RandAdjustContrastd(keys=["img"], prob=0.2, gamma=(0.5, 4.5)),
        tr.RandGaussianNoised(keys=["img"], prob=0.5, mean=0.0, std=0.1),
        tr.RandGaussianSmoothd(keys=["img"], prob=0.2),
        tr.DivisiblePadd(keys=["img", "seg"], k=8),  # can be take back to 8 if not refinement
        tr_supp.NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=True,
                                    remove_outliers=args.remove_outliers),
        tr.ToTensord(keys=["img", "seg"]),
    ])

    if all(i == 0 for i in args.patch_size):
        remove_trans = [tr.SpatialPadd, tr.RandSpatialCropd]
        train_transforms.transforms = tuple(
            [trans for trans in train_transforms.transforms if type(trans) not in remove_trans])

    if args.already_preprocess:
        remove_trans = [tr.CropForegroundd, tr.NormalizeIntensityd]
        train_transforms.transforms = tuple(
            [trans for trans in train_transforms.transforms if type(trans) not in remove_trans])

    val_transforms = tr.Compose([
        tr.LoadImaged(keys=["img", "seg"], reader="nibabelreader"),
        tr.ConvertToMultiChannelBasedOnBratsClassesd(keys=["seg"]),
        tr.CropForegroundd(keys=["img", "seg"], source_key="img"),

        # add for the distance map transform (usefull for boundary loss)
        tr.CopyItemsd(keys=["seg"], times=1, names=["distance_map"]),
        tr_supp.OneHotToDistd(keys=["distance_map"], sampling=[1, 1, 1]),

        tr_supp.NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=True,
                                    remove_outliers=args.remove_outliers),
        tr.ToTensord(keys=["img", "seg"]),
    ])

    # remove computation of distance map if not boundary loss
    if "boundary" not in args.criterion:
        remove_trans = [tr.CopyItemsd, tr_supp.OneHotToDistd]
        train_transforms.transforms = tuple(
            [trans for trans in train_transforms.transforms if type(trans) not in remove_trans])
        val_transforms.transforms = tuple(
            [trans for trans in val_transforms.transforms if type(trans) not in remove_trans])
        # also add the key for distance map when encounter seg
    else:
        train_transforms = add_key_when_distance_map(train_transforms)
        val_transforms = add_key_when_distance_map(val_transforms)

    set_determinism(seed=args.seed)
    train_transforms.set_random_state(seed=args.seed)

    # get files
    train_files, val_files = create_train_val_dataset_files(args)

    # define dataset, dataloader
    train_set = Dataset(data=train_files, transform=train_transforms) if train_files else None

    if train_in_val_mode:
        val_set = Dataset(data=train_files, transform=val_transforms) if train_files else None
    else:
        val_set = Dataset(data=val_files, transform=val_transforms) if val_files else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=list_data_collate, pin_memory=False) if train_set else None

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            collate_fn=list_data_collate, pin_memory=False) if val_set else None

    return train_loader, val_loader


def add_key_when_distance_map(transforms: tr.Compose):
    """
    Insert missing keys for distance map
    :param transforms: transforms
    :return: updated transforms
    """

    encounter_dm = False
    train_trans_added = []
    for trans in transforms.transforms:
        if "distance_map" not in trans.keys and encounter_dm is False:
            train_trans_added.append(trans)
        else:
            encounter_dm = True

            if encounter_dm is True and "seg" in trans.keys:
                trans.keys = trans.keys + ("distance_map",)
                train_trans_added.append(trans)
            else:
                train_trans_added.append(trans)
    transforms.transforms = tuple(train_trans_added)
    return transforms


def make_test_dataloader(args: argparse.Namespace) -> Union[DataLoader, None]:
    """
     get test dataloader

     :param args: ``argparse``
     :return: test_loader
     """
    test_transforms = tr.Compose([
        tr.LoadImaged(keys=["img"], reader="nibabelreader"),
        tr.CropForegroundd(keys=["img"], source_key="img"),
        tr.NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=True),
        # tr.DivisiblePadd(keys=["img"], k=8),
        tr.ToTensord(keys=["img"]),
    ])

    set_determinism(seed=args.seed)

    # get files
    test_files = create_test_dataset_files(args)

    # define dataset, dataloader
    test_set = Dataset(data=test_files, transform=test_transforms) if test_files else None

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.num_workers,
                             collate_fn=list_data_collate, pin_memory=False) if test_set else None

    return test_loader


def create_train_val_dataset_files(args: argparse.Namespace) \
        -> Tuple[Optional[List[Dict[str, Union[list, Any]]]], Optional[List[Dict[str, Union[list, Any]]]]]:
    """
    create train val dataset files

    :param args: ``argparse``
    :return: train_files, val_files
    """

    # FILTERING = ["BraTS2021_01616", "BraTS2021_01636", "BraTS2021_00810", "BraTS2021_01147", "BraTS2021_01104"]

    if args.val_data_path is not None and args.fold is not None:
        raise ValueError("Specified a val_data_path when a fold split is specified is not possible")

    train_files, val_files, test_files = None, None, None
    if args.val_data_path:
        val_dataset_dict = create_database(args.val_data_path, required_modality=["t1", "t1ce", "flair", "t2", "seg"])
        val_files = [{"patient_id": ptid, "img": list(mod.values())[:-1], "seg": list(mod.values())[-1]} for ptid, mod
                     in
                     val_dataset_dict.items()]

    if args.train_data_path:
        train_dataset_dict = create_database(args.train_data_path,
                                             required_modality=["t1", "t1ce", "flair", "t2", "seg"])
        train_id = list(train_dataset_dict)
        val_id = list()
        if args.fold is not None and not args.val_data_path:
            k_fold = KFold(5, shuffle=True, random_state=args.seed)
            splits = list(k_fold.split(list(train_dataset_dict)))
            train_idx, val_idx = splits[args.fold]
            train_id = [list(train_dataset_dict)[i] for i in train_idx]
            val_id = [list(train_dataset_dict)[i] for i in val_idx]

        train_files = [{"patient_id": ptid, "img": list(mod.values())[:-1], "seg": list(mod.values())[-1]} for ptid, mod
                       in
                       train_dataset_dict.items() if ptid in train_id]
        val_files = [{"patient_id": ptid, "img": list(mod.values())[:-1], "seg": list(mod.values())[-1]} for ptid, mod
                     in
                     train_dataset_dict.items() if ptid in val_id]

    if train_files in [None, []] and val_files in [None, []]:
        raise ValueError("train files and val files are empty")

    # add of filtering

    return train_files, val_files


def create_test_dataset_files(args: argparse.Namespace) -> List[Dict[str, Union[list, Any]]]:
    """
    create test dataset files

    :param args: ``argparse``
    :return: test_files
    """
    if hasattr(args, "docker_test") and args.docker_test:
        test_dataset_dict = create_database_test_docker(args.test_data_path,
                                                        required_modality=["t1", "t1ce", "flair", "t2"])
    else:
        test_dataset_dict = create_database(args.test_data_path, required_modality=["t1", "t1ce", "flair", "t2"])
    test_files = [{"patient_id": ptid, "img": list(mod.values())} for ptid, mod in test_dataset_dict.items()]
    return test_files


def get_tta_transforms() -> tta.Compose:
    """
    get test time augmentation transforms

    :return: transforms_tta: :py:class:`tta.Compose`
    """
    transforms_tta = tta.Compose([tta.OnAxes(axes=["zxy", "xyz"]),  # "yzx"
                                  # tta.RandomGaussianNoise(),
                                  tta.HorizontalFlip(),
                                  tta.Rotate90(angles=[0, 90, 180, 270]),
                                  ])
    return transforms_tta


def get_activation() -> tr.Compose:
    """
    get activation transform

    :return: post_transforms: :py:class:`monai.transforms.Compose`
    """
    activation_transforms = tr.Compose([tr.Activations(sigmoid=True)])
    return activation_transforms


def get_post_transforms(args: argparse.Namespace) -> tr.Compose:
    """
    get post transforms (ie after an activation, threshold if sigmoid? post processing ...)

    :param args: ``argparse``
    :return: post_transforms: :py:class:`monai.transforms.Compose`
    """

    if (hasattr(args, "replace_value") and args.replace_value) or (
            hasattr(args, "cleaning_areas") and args.cleaning_areas):
        # tr_list = [tr.AsDiscrete(argmax=True)]
        tr_list = [tr.AsDiscrete(threshold_values=True,
                                 logit_thresh=0.5 if not hasattr(args, "logit_threshold") else args.logit_threshold),
                   tr_supp.ConvertToBratsClassesBasedOnMultiChannel(),
                   tr_supp.ChangeLabel3To4()]
        if hasattr(args, "cleaning_areas") and args.cleaning_areas:
            tr_list += [tr_supp.KeepLargestConnectedComponent(threshold=args.cleaning_areas_threshold)]
        if hasattr(args, "replace_value") and args.replace_value:
            tr_list += [tr_supp.ReplaceWithClosestValue(labels=[3],
                                                        thresh=args.replace_value_threshold,
                                                        )]
        # can be accelerate because of swap numpy to tensor: ConvertToMultiChannelBasedOnBratsClasses
        tr_list += [tr.SqueezeDim(0), tr.ConvertToMultiChannelBasedOnBratsClasses(), tr.ToTensor(), tr.AddChannel()]
        post_transforms = tr.Compose(tr_list)
    else:
        post_transforms = tr.Compose([tr.AsDiscrete(threshold_values=True, logit_thresh=0.5 if not hasattr(args,
                                                                                                           "logit_threshold") else args.logit_threshold)])
    return post_transforms


def get_save_seg_transforms() -> tr.Compose:
    """
    get save seg transforms (ie pass to one hot format, to binarize ...)

    :return: post_transforms: :py:class:`monai.transforms.Compose`
    """
    save_seg_transforms = tr.Compose([tr_supp.ConvertToBratsClassesBasedOnMultiChannel(),
                                      tr_supp.ChangeLabel3To4()])
    return save_seg_transforms
