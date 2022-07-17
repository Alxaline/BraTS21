# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Nov 23, 2020
Adapted from: https://github.com/qubvel/ttach
"""
from typing import List, Sequence

import torch
from monai.transforms.compose import Randomizable
from torch import rot90

from .base import DualTransform, ImageOnlyTransform


class OnAxes(DualTransform):
    """
    For 3D images will permute the axe to compose in the different plane: ["xyz", "yzx", "zxy"]
    """

    identity_param = "zxy"

    def __init__(self, axes: List[str]):
        """
        Args:
            axes: list of axe ["xyz", "yzx", "zxy"]
        """
        super().__init__("axe", axes)

        assert all(axe in ["xyz", "yzx", "zxy"] for axe in axes), "axes need to be 'xyz', 'yzx', 'zxy'"

    # xyz, yzx, zxy
    def apply_aug_image(self, image: torch.Tensor, axe: str = "zxy", **kwargs) -> torch.Tensor:
        if axe == "zxy":
            return image
        elif axe == "xyz":
            return image.permute(0, 1, 3, 4, 2)
        elif axe == "yzx":
            return image.permute(0, 1, 4, 2, 3)

    def apply_deaug_mask(self, mask: torch.Tensor, axe: str = "zxy", **kwargs) -> torch.Tensor:
        if axe == "zxy":
            return mask
        elif axe == "xyz":
            return mask.permute(0, 1, 4, 2, 3)
        elif axe == "yzx":
            return mask.permute(0, 1, 3, 4, 2)

    def apply_deaug_label(self, label: torch.Tensor, **kwargs) -> torch.Tensor:
        return label


class HorizontalFlip(DualTransform):
    """
    Flip images horizontally (left->right)
    """

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image: torch.Tensor, apply: bool = False, **kwargs) -> torch.Tensor:
        if apply:
            image = image.flip(3)
        return image

    def apply_deaug_mask(self, mask: torch.Tensor, apply: bool = False, **kwargs) -> torch.Tensor:
        if apply:
            mask = mask.flip(3)
        return mask

    def apply_deaug_label(self, label: torch.Tensor, **kwargs) -> torch.Tensor:
        return label


class VerticalFlip(DualTransform):
    """
    Flip images vertically (up->down)
    """

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image: torch.Tensor, apply: bool = False, **kwargs) -> torch.Tensor:
        if apply:
            image = image.flip(2)
        return image

    def apply_deaug_mask(self, mask: torch.Tensor, apply: bool = False, **kwargs) -> torch.Tensor:
        if apply:
            mask = mask.flip(2)
        return mask

    def apply_deaug_label(self, label: torch.Tensor, **kwargs):
        return label


class RandomGaussianNoise(ImageOnlyTransform, Randomizable):
    """
    Add Random Gaussian Noise to image with mean 0 and std 0.1
    """

    identity_param = True

    def __init__(self):
        self.mean = 0.0
        self.std = 0.1
        self._do_transform = True
        self._noise = None

        super().__init__("apply", [True])

    def randomize(self, im_shape: Sequence[int]) -> None:
        self._noise = self.R.normal(self.mean, self.R.uniform(0, self.std), size=im_shape)

    def apply_aug_image(self, image: torch.Tensor, **kwargs):
        self.randomize(image.shape)
        device = image.get_device()
        return image + torch.from_numpy(self._noise).type(image.dtype).to(device)


class GaussianNoise(DualTransform):
    """
    Flip images vertically (up->down)
    """

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image: torch.Tensor, apply: bool = False, **kwargs) -> torch.Tensor:
        if apply:
            image = image.flip(2)
        return image

    def apply_deaug_mask(self, mask: torch.Tensor, apply: bool = False, **kwargs) -> torch.Tensor:
        if apply:
            mask = mask.flip(2)
        return mask

    def apply_deaug_label(self, label: torch.Tensor, **kwargs) -> torch.Tensor:
        return label


class Rotate90(DualTransform):
    """
    Rotate images 0/90/180/270 degrees
    """

    identity_param = 0

    def __init__(self, angles: List[int]):
        """
        Args:
            angles: list of angle [0, 90, 180, 270]
        """
        if self.identity_param not in angles:
            angles = [self.identity_param] + list(angles)
        super().__init__("angle", angles)

    def apply_aug_image(self, image: torch.Tensor, angle: int = 0, **kwargs) -> torch.Tensor:
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return rot90(image, k, (2, 3))

    def apply_deaug_mask(self, mask: torch.Tensor, angle: int = 0, **kwargs) -> torch.Tensor:
        return self.apply_aug_image(mask, -angle)

    def apply_deaug_label(self, label: torch.Tensor, angle: int = 0, **kwargs) -> torch.Tensor:
        return label
