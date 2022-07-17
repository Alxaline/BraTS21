# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import logging
import os
import resource
import time

import torch

from learning.engine import Engine
from src import set_main_logger
from src.arguments_inference import get_args
from src.definer import make_train_val_dataloader, make_test_dataloader, get_model, make_criterion, get_post_transforms, \
    get_activation, get_save_seg_transforms

# avoid this issues: https://github.com/Project-MONAI/MONAI/issues/701
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def main() -> None:
    """
    Main function to do inference
    """
    total_start = time.time()

    config_dict, experience_name = get_args()

    # verify cuda is available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config_dict["config_0"].device
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"

    # create save_path
    if not os.path.exists(config_dict["config_0"].save_path):
        os.makedirs(config_dict["config_0"].save_path, exist_ok=True)

    # set logger
    set_main_logger(log_file=True, filename=os.path.join(config_dict["config_0"].save_path, "logfile_inference.log"),
                    verbosity_lvl=config_dict["config_0"].verbosity)
    logger = logging.getLogger(__name__)

    valid_loader, test_loader = None, None
    # define dataloader (determinism is set inside function)
    if config_dict["config_0"].on == "train":
        _, valid_loader = make_train_val_dataloader(config_dict["config_0"], train_in_val_mode=True)
    elif config_dict["config_0"].on == "val":
        _, valid_loader = make_train_val_dataloader(config_dict["config_0"])
    elif config_dict["config_0"].on == "test":
        test_loader = make_test_dataloader(config_dict["config_0"])

    # define loss criterion
    criterion = make_criterion(config_dict["config_0"]).cuda()

    current_time = time.time()
    logger.info(f"Time elapsed before Inference: {current_time - total_start}")

    # set up engine
    model_list = []
    for _, args in config_dict.items():
        model = get_model(args)
        model = torch.optim.swa_utils.AveragedModel(model) if args.swa_start else model
        model_type = "swa_model" if args.swa_start else "model"
        model_name_to_load = "last_model.pth" if args.swa_start else "best_model.pth"
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(args.config), model_name_to_load))[model_type])
        model_list.append(model)

    suffix_step_mode = f"_{config_dict['config_0'].on}"
    engine = Engine(model_list, criterion, config_dict["config_0"].num_classes, None,
                    key_metric=config_dict["config_0"].key_metric,
                    additional_metrics=config_dict["config_0"].additional_metrics)

    engine.evaluate(valid_loader if config_dict["config_0"].on in ["val", "train"] else test_loader,
                    0,
                    config_dict["config_0"],
                    True if config_dict["config_0"].tta else False,
                    False,
                    get_post_transforms(config_dict["config_0"]),
                    get_activation(),
                    True,
                    get_save_seg_transforms(),
                    os.path.join(config_dict["config_0"].save_path,
                                 # f"Inference_segmentations{suffix_step_mode}{experience_name}"
                                 ),
                    None,
                    f"Evaluation_inference{suffix_step_mode}{experience_name}",
                    f"metric_{config_dict['config_0'].on}",
                    True if config_dict["config_0"].on in ["val", "train"] else False,
                    True)


if __name__ == "__main__":
    main()
