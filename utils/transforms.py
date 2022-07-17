# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE (alexandre.carre@gustaveroussy.fr)
| Created on: Jan 02, 2021
"""
import logging
from itertools import chain
from typing import Dict, Hashable, Mapping, Tuple, Union, Sequence, Optional

import SimpleITK as sitk
import numpy as np
import torch
from monai.config import DtypeLike
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform, BorderPad
from scipy import interpolate
from scipy.ndimage import distance_transform_edt as edt
from skimage import morphology
from torch.nn import functional as F

from utils.misc import tensor_to_array

logger = logging.getLogger(__name__)


class OneHotd(MapTransform):
    """
    For a array `labels` of dimensions B1[spatial_dims], return a array of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.
    """

    def __init__(self, keys: KeysCollection, num_classes: int) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            num_classes: number o classes
        """
        super().__init__(keys)

        self.num_classes = num_classes

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = np.eye(self.num_classes)[d[key]]
        return d


class ConvertLabel(Transform):
    """
    Convert Label : can be useful for brats where the label are 1,2,4 and can be change in 1,2,3 or
    simply for merging label
    """

    def __init__(self, in_label: Sequence[int], out_label: Sequence[int]):
        """
        Args:
            in_label (list, tuple): the entry label
            out_label (list, tuple): the output label
        """
        self.in_label = in_label
        self.out_label = out_label

    def __call__(self, data: np.ndarray) -> np.ndarray:
        convert_volume = np.copy(data)
        for i in range(len(self.in_label)):
            convert_volume[data == self.in_label[i]] = self.out_label[i]
        return convert_volume.astype('int8')


class ConvertLabeld(MapTransform):
    """
    Convert Label : can be useful for brats where the label are 1,2,4 and can be change in 1,2,3 or
    simply for merging label
    Args:
        keys (list): parameter will be used to get and set the actual data item to transform.
        in_label (list, tuple): the entry label
        out_label (list, tuple): the output label
    """

    def __init__(self, keys, in_label: Sequence[int], out_label: Sequence[int]):
        super().__init__(keys)
        assert len(in_label) == len(out_label), "Length of in_label and out label must be the same"

        self.convert_label = ConvertLabel(in_label, out_label)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.convert_label(d[key])
        return d


class OneHotToDist(Transform):
    """
    One hot to distance Map.
    Idea is to compute the distance map as https://github.com/LIVIAETS/boundary-loss
    """

    def __init__(self, sampling: Sequence[int]):
        """
        Args:
            sampling (list, tuple): resolution sampling
        """
        self.sampling = sampling

    def __call__(self, data: np.ndarray) -> np.ndarray:

        K: int = len(data)

        res = np.zeros_like(data)
        for k in range(K):
            posmask = data[k].astype(bool)
            if posmask.any():
                negmask = ~posmask
                res[k] = edt(negmask, sampling=self.sampling) * negmask - (
                        edt(posmask, sampling=self.sampling) - 1) * posmask
            # The idea is to leave blank the negative classes
            # since this is one-hot encoded, another class will supervise that pixel

        return res


class OneHotToDistd(MapTransform):
    """
    One hot to distance Map
    Args:
        keys (list): parameter will be used to get and set the actual data item to transform.
        sampling (list, tuple): resolution sampling
    """

    def __init__(self, keys, sampling: Sequence[int]):
        super().__init__(keys)

        self.one_hot_to_dist = OneHotToDist(sampling)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.one_hot_to_dist(d[key])
        return d


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the necrotic and non-enhancing tumor core (NCR/NET)
    label 2 is the peritumoral edema (ED)
    label 4 is the GD-enhancing tumor (ET)
    The possible classes are WT (Whole tumor = ED + NCR/NET + ET), TC (Tumor core = ET + NCR/NET),
    and ET (Enhancing tumor = ET).
    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            result = list()
            # merge labels 1, 2 and 4 to construct WT
            result.append(np.logical_or(np.logical_or(d[key] == 1, d[key] == 4), d[key] == 2))
            # merge label 1 and label 4 to construct TC
            result.append(np.logical_or(d[key] == 1, d[key] == 4))
            # label 4 is ET
            result.append(d[key] == 4)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class ConvertToBratsClassesBasedOnMultiChannel(Transform):
    """
    Convert multi channels (one hot data) to labels corresponding to brats classes:
    The channel are WT (Whole tumor = ED + NCR/NET + ET), TC (Tumor core = ET + NCR/NET),
    and ET (Enhancing tumor = ET).
    label 1 is the necrotic and non-enhancing tumor core (NCR/NET)
    label 2 is the peritumoral edema (ED)
    label 3 is the GD-enhancing tumor (ET)
    return BNCHW where N is axis order [1,2,3]
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndimension() == 5, "number of dimension is incorrect, need to be BNCHW"
        assert data.shape[0] == 1, "Batch dim need to be 1"
        assert data.shape[1] == 3, "Number of channel need to be 3 (TC/WT/ET)"
        # array = tensor_to_array(data)
        et = data[0][2].bool()
        net = torch.logical_and(data[0][0], torch.logical_not(et))
        ed = torch.logical_and(data[0][1], torch.logical_not(data[0][0]))
        label_map = torch.zeros(data[0][0].shape)
        label_map[et] = 3
        label_map[net] = 1
        label_map[ed] = 2
        label_map = label_map[None, None]
        return label_map


class ChangeLabel3To4(Transform):
    """
    Change the label 3 to 4 for ET to corresponds to BraTS label
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndimension() == 5, "number of dimension is incorrect, need to be BNCHW"
        assert data.shape[0] == 1, "Batch dim need to be 1"
        assert data.shape[1] == 1, "Channel need to be 1"
        data[data == 3] = 4
        return data


class KeepLargestConnectedComponent(Transform):
    """
    Keeps only the largest connected component in the image (or above a threshold).
    This transform can be used as a post-processing step to clean up over-segment areas in model output.
    """

    def __init__(self, threshold: Optional[int] = None) -> None:
        """
        Args:
            threshold: threshold to keep labels connected components. If None keep the largest one
        """

        self.threshold = threshold

    def __call__(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        if torch.is_tensor(data):
            device = data.device
            data = tensor_to_array(data)
            return torch.as_tensor(get_largest_component(data, threshold=self.threshold), dtype=torch.float32,
                                   device=device)

        return get_largest_component(data, threshold=self.threshold)


class ReplaceWithClosestValue(Transform):
    """
    Replace values with the closest one in a 3D array following a specific axis.
    Execute after model forward and when value are already binarized.

    Args:
        labels: labels values to replace with the closest one.
        n_classes: number of class to convert to one hot format
        thresh: the threshold value for thresholding operation. (default to 20)
        axis: axis to realize the interpolation in 2D. default (axis=2, axial)
    """

    def __init__(
            self,
            labels: Sequence[int],
            thresh: int = 20,
            axis: int = 2,
    ) -> None:
        self.labels = labels
        self.thresh = thresh
        self.axis = axis

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        assert data.ndimension() == 5 and data.shape[1] == 1 and data.shape[0] == 1, "data shape must be 11HWD"

        data = data.squeeze()
        array = tensor_to_array(data)

        label_unique, label_counts = np.unique(array, return_counts=True)
        logger.debug(f"label count: {label_counts}")

        values_to_replace = label_unique[label_counts <= self.thresh]
        if values_to_replace.any():
            data = replace_w_closest_value_3d(input_array=array, values=values_to_replace, axis=self.axis)

        return data[None, None]


class MinMaxScalerIntensity(Transform):
    """
    Normalize input based on provided args, using Basic min max scaler.
    Args:
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
            self,
            nonzero: bool = True,
            channel_wise: bool = True,
            low_perc: int = 1,
            high_perc: int = 99,
            clip: bool = True,

    ) -> None:
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.low_perc = low_perc
        self.high_perc = high_perc
        self.clip = clip

    def _normalize(self, img):

        if self.nonzero:
            non_zeros = img > 0
        else:
            non_zeros = np.ones(img.shape, dtype=bool)

        if self.clip:
            low, high = np.percentile(img[non_zeros], [self.low_perc, self.high_perc])
            # Perform clipping in non_zero values
            img[img > high] = high
            img[(img < low) & (img > 0)] = low

        min_ = np.min(img)
        max_ = np.max(img)
        scale = max_ - min_
        img = (img - min_) / scale

        return img

    def __call__(self, img):
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        if self.channel_wise:
            for i, d in enumerate(img):
                img[i] = self._normalize(d)
        else:
            img = self._normalize(img)

        return img


class NormalizeIntensity(Transform):
    """
    Normalize input based on provided args, using calculated mean and std if not provided.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    When `channel_wise` is True, the first dimension of `subtrahend` and `divisor` should
    be the number of image channels if they are not None.

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
        dtype: output data type, defaults to float32.
    """

    def __init__(
            self,
            subtrahend: Union[Sequence, np.ndarray, None] = None,
            divisor: Union[Sequence, np.ndarray, None] = None,
            nonzero: bool = False,
            channel_wise: bool = False,
            dtype: DtypeLike = np.float32,
            remove_outliers: bool = False,
            outliers_value: float = 3,

    ) -> None:
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise
        self.dtype = dtype
        self.remove_outliers = remove_outliers
        self.outliers_value = outliers_value

    def _normalize(self, img: np.ndarray, sub=None, div=None) -> np.ndarray:
        slices = (img != 0) if self.nonzero else np.ones(img.shape, dtype=bool)
        if not np.any(slices):
            return img

        _sub = sub if sub is not None else np.mean(img[slices])
        if isinstance(_sub, np.ndarray):
            _sub = _sub[slices]

        _div = div if div is not None else np.std(img[slices])
        if np.isscalar(_div):
            if _div == 0.0:
                _div = 1.0
        elif isinstance(_div, np.ndarray):
            _div = _div[slices]
            _div[_div == 0.0] = 1.0
        img[slices] = (img[slices] - _sub) / _div

        if self.remove_outliers:
            img[slices] = np.clip(img[slices], - self.outliers_value, self.outliers_value)

        return img

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`, assuming `img` is a channel-first array if `self.channel_wise` is True,
        """
        if self.channel_wise:
            if self.subtrahend is not None and len(self.subtrahend) != len(img):
                raise ValueError(f"img has {len(img)} channels, but subtrahend has {len(self.subtrahend)} components.")
            if self.divisor is not None and len(self.divisor) != len(img):
                raise ValueError(f"img has {len(img)} channels, but divisor has {len(self.divisor)} components.")

            for i, d in enumerate(img):
                img[i] = self._normalize(
                    d,
                    sub=self.subtrahend[i] if self.subtrahend is not None else None,
                    div=self.divisor[i] if self.divisor is not None else None,
                )
        else:
            img = self._normalize(img, self.subtrahend, self.divisor)

        return img.astype(self.dtype)


class NormalizeIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        subtrahend: the amount to subtract by (usually the mean)
        divisor: the amount to divide by (usually the standard deviation)
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
        dtype: output data type, defaults to float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
            self,
            keys: KeysCollection,
            subtrahend: Optional[np.ndarray] = None,
            divisor: Optional[np.ndarray] = None,
            nonzero: bool = False,
            channel_wise: bool = False,
            remove_outliers: bool = False,
            outliers_value: float = 3,
            dtype: DtypeLike = np.float32,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.normalizer = NormalizeIntensity(subtrahend, divisor, nonzero, channel_wise, dtype, remove_outliers,
                                             outliers_value)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.normalizer(d[key])
        return d


class MinMaxScalerIntensityd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.NormalizeIntensity`.
    This transform can normalize only non-zero values or entire image, and can also calculate
    mean and std on each channel separately.
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        nonzero: whether only normalize non-zero values.
        channel_wise: if using calculated mean and std, calculate on each channel separately
            or calculate on the entire image directly.
    """

    def __init__(
            self,
            keys: KeysCollection,
            nonzero: bool = True,
            channel_wise: bool = True,
            low_perc: int = 1,
            high_perc: int = 99,
            clip: bool = True,

    ) -> None:
        super().__init__(keys)
        self.normalizer = MinMaxScalerIntensity(nonzero, channel_wise, low_perc, high_perc, clip)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.normalizer(d[key])
        return d


def shape_to_divisible(data: torch.Tensor, k: int = 16, min_shape: int = None) \
        -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Pad the input data, so that the spatial sizes are divisible by k.

    :param data: input tensor
    :param k: the target k for each spatial dimension.
    :param min_shape: min shape to return for all dimensions
    :returns tensor shape to model, p_b, p_a
    """
    assert k > 0, 'k need to positive'

    # consider channel first
    if data.ndim == 5:
        data_shape = data.shape[2:]
    elif data.ndim == 4:
        data_shape = data.shape[1:]
    else:
        raise ValueError("Tensor dimension is incorrect")

    zero_pad_shape = np.ceil(np.divide(data_shape, k)).astype(np.int) * k
    if min_shape is not None:
        zero_pad_shape[zero_pad_shape < min_shape] = min_shape

    p = zero_pad_shape - data_shape  # padding
    p_b = np.ceil(p / 2).astype(np.int)  # padding before image
    p_a = np.floor(p / 2).astype(np.int)  # padding after image
    data_pad = F.pad(data, (p_b[2], p_a[2], p_b[1], p_a[1], p_b[0], p_a[0]), mode='constant', value=0)

    return data_pad, p_b, p_a


def shape_to_original(data: torch.Tensor, p_b: np.ndarray, p_a: np.ndarray) -> torch.Tensor:
    """
    From a random tensor used in the function shape_to_model, return the tensor with original shape.

    :param data: tensor
    :param p_b: n-D array
    :param p_a: n-D array
    :return tensor with original shape
    """

    # consider channel first
    if data.ndim == 5:
        datashape = data.shape[2:]
    elif data.ndim == 4:
        datashape = data.shape[1:]
    else:
        raise ValueError("Tensor dimension is incorrect")
    p_up = datashape - p_a
    return data[..., p_b[0]:p_up[0], p_b[1]:p_up[1], p_b[2]:p_up[2]].contiguous()


def remove_background_voxels(img: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """
    Remove potential voxels in background

    :param img: image tensor
    :param outputs: outputs pred tensor
    :return: outputs tensor with removed background voxels
    """
    # remove potential voxels in background
    brain_binary_mask = torch.clone(img).to(img.get_device())
    brain_binary_mask[brain_binary_mask != 0] = 1
    brain_binary_mask = torch.sum(brain_binary_mask, dim=1)
    brain_binary_mask[brain_binary_mask != 0] = 1
    outputs = outputs * brain_binary_mask
    return outputs


def pad_back_to_shape_before_compose(batch_data: torch.Tensor, outputs: Union[torch.tensor, np.ndarray]) -> np.ndarray:
    """
    Pad back to the original shape before apply monai.transforms.Compose

    :param batch_data: batch data from the data loader
    :param outputs: outputs pred tensor
    :return: return outputs n-D array with the original shape
    """

    outputs = tensor_to_array(outputs)

    if outputs.ndim == 5 and outputs.shape[0] == 1:
        outputs = outputs[0]
    assert outputs.ndim == 4, "data shape must be NCHW"

    border_pad_start = tensor_to_array(batch_data["foreground_start_coord"]).squeeze()
    border_pad_end = [a_i - b_i for a_i, b_i in zip(tensor_to_array(batch_data["img_meta_dict"]["spatial_shape"][0]),
                                                    tensor_to_array(batch_data["foreground_end_coord"]).squeeze())]

    spatial_border = list(chain.from_iterable(zip(border_pad_start, border_pad_end)))
    spatial_border = list(map(int, spatial_border))
    pad_back = BorderPad(spatial_border=spatial_border)
    outputs = pad_back(outputs)[0]
    return outputs


def get_largest_component(in_volume: np.ndarray,
                          threshold: Optional[int] = None,
                          ):
    """
    Look for connected component and choose the largest one if threshold is none, else choose connected components
    above threshold.

    :param in_volume: data n-D array (seg)
    :param threshold: threshold to keep labels connected components
    :return: n-D array with label change
    """
    # basically look for connected components and choose the largest one, delete everything else
    mask = in_volume != 0
    lbls = morphology.label(mask)
    lbls_sizes = [np.sum(lbls == i) for i in np.unique(lbls)]
    if threshold is None:
        region = np.argmax(lbls_sizes[1:]) + 1
    else:
        region = np.argwhere(np.array(lbls_sizes[1:]) > threshold).squeeze() + 1

    in_volume[~np.isin(lbls, region)] = 0
    return in_volume


def replace_w_closest_value_2d(input_array: np.ndarray, values: Sequence[int]) -> np.ndarray:
    """
    Replace values with the closest one in a 2D array.

    :param input_array: 2D array
    :param values: values to replace with closest
    """
    assert len(input_array.shape) == 2, 'input_array need to be a 2D array'

    array = np.ma.MaskedArray(input_array, np.in1d(input_array, values))

    if np.any(array.mask):
        x = np.arange(0, input_array.shape[1])
        y = np.arange(0, input_array.shape[0])
        xx, yy = np.meshgrid(x, y)
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]
        arr_interpolate = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')
        return arr_interpolate
    else:
        return array


def replace_w_closest_value_3d(input_array: np.ndarray, values: Sequence[int], axis: int) -> np.ndarray:
    """
    Replace values with the closest one in a 3D array following a specific axis.

    :param input_array: 3D array
    :param values:  values to replace with closest
    :param axis: axis were to interpolate '0, 1, 2'
    """

    assert len(input_array.shape) == 3, 'Need to be 3D array'
    assert 0 <= axis <= 2, 'axis need to be 0, 1 or 2'
    new_arr = np.zeros(input_array.shape, dtype=np.uint8)

    for i in range(input_array.shape[axis]):
        if axis == 0:
            new_arr[i] = replace_w_closest_value_2d(input_array=input_array[i], values=values)
        elif axis == 1:
            new_arr[:, i] = replace_w_closest_value_2d(input_array=input_array[:, i, :], values=values)
        else:
            new_arr[:, :, i] = replace_w_closest_value_2d(input_array=input_array[:, :, i], values=values)
    return new_arr


def perform_staple_on_brats_multi_channel(datas: Sequence[Union[np.ndarray, torch.Tensor]],
                                          threshold_value: float = 0.50,
                                          return_as_tensor: bool = True):
    """
    For the moment only BS1 is supported
    Perform STAPLE algorithm in a binary way for each brats class:
    Channel 0 = TC
    Channel 1 = WT
    Channel 2 = ET

    The STAPLE algorithm is described in:
    S. Warfield, K. Zou, W. Wells, "Validation of image segmentation and expert quality with an expectation-maximization
    algorithm" in MICCAI 2002: Fifth International Conference on Medical Image Computing and Computer-Assisted
    Intervention, Springer-Verlag, Heidelberg, Germany, 2002, pp. 298-306

    :param datas: list of array or tensor with shape [1x3xH,W,D]
    :param threshold_value: threshold value for binarization of staple probabilities
    :param return_as_tensor: if true return a tensor
    """
    datas = [d.detach().cpu().numpy() for d in datas if torch.is_tensor(d)]
    datas = [d.astype('uint8') for d in datas]

    staple_filter = sitk.STAPLEImageFilter()
    staple_filter.SetMaximumIterations(10000)
    staple_filter.SetForegroundValue(1.0)

    # first index is BS
    tc = sitk.GetArrayFromImage(
        staple_filter.Execute([sitk.GetImageFromArray(d[0][0]) for d in datas]) > threshold_value)
    wt = sitk.GetArrayFromImage(
        staple_filter.Execute([sitk.GetImageFromArray(d[0][1]) for d in datas]) > threshold_value)
    et = sitk.GetArrayFromImage(
        staple_filter.Execute([sitk.GetImageFromArray(d[0][2]) for d in datas]) > threshold_value)

    stapled_data = np.stack([tc, wt, et])[None]  # add back BS
    if return_as_tensor:
        return torch.from_numpy(stapled_data)
    return stapled_data
