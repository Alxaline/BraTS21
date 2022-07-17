# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import argparse
import os
from argparse import Namespace
from copy import deepcopy
from typing import Dict, Tuple

import oyaml as yaml

from src.definer import make_experience_name
from utils.files import check_exist, check_isdir


def add_inference_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Inference args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('model', 'Model configuration')
    group.add_argument("--config", type=check_exist, required=True, default=None, nargs="+",
                       help="path(s) to the trained models config yaml file you want to use")
    group.add_argument("--train_data_path", type=check_isdir, help="path to the training data")
    group.add_argument("--val_data_path", type=check_isdir, default=None, help="path to the val data")
    group.add_argument("--input", "--test_data_path", dest="test_data_path", type=check_isdir, default=None,
                       help="path to the test data")
    group.add_argument('--on', default="val", choices=["val", "train", "test"])
    group.add_argument("--device", type=str, default='0', help="device id for GPU")
    group.add_argument("--output", "--save_path",  dest="save_path", type=str, default=None)
    group.add_argument("--create_patient_dir", action="store_true", default=False, help="Create patient directory")
    group.add_argument("--docker_test", action="store_true", default=False, help="Create loader for docker run")
    group.add_argument("--num_workers", type=int, default=0)
    group.add_argument("-v", "--verbosity", action="count", default=0,
                       help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def add_processing_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Processing args

    :param parser: parser
    :return: parser
    """
    group = parser.add_argument_group('processing', 'Processing configuration')
    group.add_argument("--sliding_window_inference", action="store_true", default=False,
                       help="Do sliding window for inference in validation")
    group.add_argument("--sliding_window_size", type=int, nargs=3, default=[128, 128, 128],
                       help="Roi size of the sliding window")
    group.add_argument('--tta', action="store_true", default=False)
    group.add_argument('--logit_threshold', type=float, default=0.5)
    group.add_argument('--perform_staple', action="store_true", default=False,
                       help="Perform STAPLE for fusion if several models")
    group.add_argument('--staple_threshold', type=float, default=0.5,
                       help="STAPLE threshold for binarization of probabilities (must be comprise between 0 and 1)")
    group.add_argument("--cleaning_areas", action="store_true", default=False,
                       help="Post-processing on labels. Clean labels based on numbers of connected components. "
                            "Remove all connected components areas below the threshold")
    group.add_argument("--cleaning_areas_threshold", type=int, default=10,
                       help="Value under the label will be post-processed (default 10)")
    group.add_argument("--replace_value", action="store_true", default=False,
                       help="Post-processing on labels ET. Change labels below the threshold based on "
                            "an 2D interpolation along the axial plane (ie replace with nearest label value)")
    group.add_argument("--replace_value_threshold", type=int, default=20,
                       help="Value under the label will be post-processed (default 20)")
    return parser


def get_args() -> Tuple[Dict[str, Namespace], str]:
    """
    Parse all the args

    :return: Dict of config_nb as key and argparse as value
    """
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Model Inference')

    parser = add_inference_args(parser)
    parser = add_processing_args(parser)
    args = parser.parse_args()

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    if args.on == "test":
        assert args.test_data_path is not None, "if 'on' is test, a test_data_path is needed"

    if len(args.config) > 1 and args.save_path is None:
        raise ValueError("Several configs/models files has been selected and you don't have specified a 'save_path'")

    if len(args.config) > 1:
        if args.on == "val":
            if args.on == "val" and args.val_data_path is None:
                raise ValueError("'on' is val. You need to specify a 'val_data_path' ")
        elif args.on == "train":
            raise ValueError("'on' is train. This mode is incompatible with several configs/models files ")

    config_dict = {}
    for idx_config, config_file in enumerate(args.config):
        folder = os.path.dirname(config_file)
        checkpoint_file = [f for f in os.listdir(folder) if
                           os.path.isfile(os.path.join(folder, f)) and f.endswith(".pth") and "best_model" in f]
        assert len(checkpoint_file) == 1, f"Number of checkpoint.pth found in {folder} is {len(checkpoint_file)}. " \
                                          f"Only one model is required in folder"

        with open(config_file, "r") as infile:
            loaded_config_file = yaml.safe_load(infile)

        train_data_path_from_config = deepcopy(loaded_config_file["train_data_path"])
        val_data_path_from_config = deepcopy(loaded_config_file["val_data_path"])
        save_path_from_config = deepcopy(loaded_config_file["save_path"])

        loaded_config_file.update(vars(args))
        loaded_config_file["train_data_path"] = train_data_path_from_config if loaded_config_file[
                                                                                   "train_data_path"] is None else \
            loaded_config_file["train_data_path"]
        loaded_config_file["val_data_path"] = val_data_path_from_config if loaded_config_file[
                                                                               "val_data_path"] is None else \
            loaded_config_file["val_data_path"]
        loaded_config_file["save_path"] = save_path_from_config if loaded_config_file[
                                                                       "save_path"] is None else \
            loaded_config_file["save_path"]
        loaded_config_file["no_tensorboard"] = True
        loaded_config_file["log_train_metrics"] = True
        loaded_config_file["config"] = loaded_config_file["config"][idx_config]
        loaded_config_file["model_pth"] = os.path.join(folder, checkpoint_file[0])
        loaded_config_file["log_val_interval"] = 1

        # verify data path after merge
        if args.on == "train":
            check_isdir(loaded_config_file["train_data_path"])
        elif args.on == "val":
            if args.val_data_path is not None:
                loaded_config_file["fold"] = None
                loaded_config_file["train_data_path"] = None
            if loaded_config_file["fold"] is not None:
                check_isdir(loaded_config_file["train_data_path"])
            else:
                check_isdir(loaded_config_file["val_data_path"])
        elif args.on == "test":
            check_isdir(loaded_config_file["test_data_path"])
        else:
            raise NotImplementedError

        config_dict[f"config_{idx_config}"] = argparse.Namespace(**loaded_config_file)

    experience_name, _ = make_experience_name({"processing": arg_groups["processing"]},
                                              add_current_date_time=False)

    return config_dict, experience_name
