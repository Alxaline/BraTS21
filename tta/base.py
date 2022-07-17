# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Nov 23, 2020
| Adapted from: `<https://github.com/qubvel/ttach>`_
"""
import abc
import itertools
from abc import ABC
from functools import partial
from typing import List, Union


class BaseTransform(ABC):
    """
    An abstract class of a ``BaseTransform``.
    """

    identity_param = None

    def __init__(
            self,
            name: str,
            params: Union[list, tuple],
    ):
        self.params = params
        self.pname = name

    @abc.abstractmethod
    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError

    @abc.abstractmethod
    def apply_deaug_label(self, label, *args, **params):
        raise NotImplementedError


class DualTransform(BaseTransform, ABC):
    """
    Dual transform (image and mask/label)
    """
    pass


class ImageOnlyTransform(BaseTransform, ABC):
    """
    Image only transform
    """

    def apply_deaug_mask(self, mask, *args, **params):
        return mask

    def apply_deaug_label(self, label, *args, **params):
        return label


class Chain:
    """
    Chain a series of calls together in a sequence
    """

    def __init__(
            self,
            functions: List[callable]
    ):
        self.functions = functions or []

    def __call__(self, x):
        for f in self.functions:
            x = f(x)
        return x


class Transformer:
    """
     A transform is callable that processes ``data``.
    """

    def __init__(
            self,
            image_pipeline: Chain,
            mask_pipeline: Chain,
            label_pipeline: Chain
    ):
        self.image_pipeline = image_pipeline
        self.mask_pipeline = mask_pipeline
        self.label_pipeline = label_pipeline

    def augment_image(self, image):
        return self.image_pipeline(image)

    def deaugment_mask(self, mask):
        return self.mask_pipeline(mask)

    def deaugment_label(self, label):
        return self.label_pipeline(label)


class Compose:
    """
        ``Compose`` provides the ability to chain a series of calls together in a
        sequence. Each transform in the sequence must take a single argument and
        return a single value, so that the transforms can be called in a chain.
    """

    def __init__(
            self,
            transforms: List[BaseTransform],
    ):
        self.aug_transforms = transforms
        self.aug_transform_parameters = list(itertools.product(*[t.params for t in self.aug_transforms]))
        self.deaug_transforms = transforms[::-1]
        self.deaug_transform_parameters = [p[::-1] for p in self.aug_transform_parameters]

    def __iter__(self) -> Transformer:
        for aug_params, deaug_params in zip(self.aug_transform_parameters, self.deaug_transform_parameters,
                                            ):
            image_aug_chain = Chain([partial(t.apply_aug_image, **{t.pname: p})
                                     for t, p in zip(self.aug_transforms, aug_params)])
            mask_deaug_chain = Chain([partial(t.apply_deaug_mask, **{t.pname: p})
                                      for t, p in zip(self.deaug_transforms, deaug_params)])
            label_deaug_chain = Chain([partial(t.apply_deaug_label, **{t.pname: p})
                                       for t, p in zip(self.deaug_transforms, deaug_params)])

            yield Transformer(
                image_pipeline=image_aug_chain,
                mask_pipeline=mask_deaug_chain,
                label_pipeline=label_deaug_chain,
            )

    def __len__(self) -> int:
        return len(self.aug_transform_parameters)
