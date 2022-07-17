import warnings
from typing import Callable, List, Optional, Union, Sequence
from typing import Set, Tuple, cast

import numpy as np
import torch
import torch.sparse
from monai.losses.dice import DiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction
from scipy.ndimage import distance_transform_edt as eucl_distance
from torch import Tensor
from torch import einsum
from torch.nn.modules.loss import _Loss

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    b, *img_shape = seg.shape  # type: Tuple[int, ...]

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    return res


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape

    res = probs.argmax(dim=1)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape

    res = class2one_hot(probs2class(probs), K)

    return res


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                     - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def one_hot2hd_dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                    dtype=None) -> np.ndarray:
    """
    Used for https://arxiv.org/pdf/1904.10030.pdf,
    implementation from https://github.com/JunMa11/SegWithDistMap
    """
    # Relasx the assertion to allow computation live on only a
    # subset of the classes
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            res[k] = eucl_distance(posmask, sampling=resolution)

    return res


class HausdorffLoss(_Loss):
    """
    Adapted from https://github.com/JunMa11/SegWithDistMap which
    Implementation heavily inspired from https://github.com/JunMa11/SegWithDistMap
    """

    def __init__(self,
                 idc,
                 alpha=2.0,
                 to_onehot_y: bool = False,
                 sigmoid: bool = False,
                 softmax: bool = False,
                 other_act: Optional[Callable] = None,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN,
                 ):
        super().__init__(reduction=LossReduction(reduction).value)

        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = idc
        self.alpha = alpha
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        print(f"Initialized {self.__class__.__name__} with {idc}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:

        if self.sigmoid:
            probs = torch.sigmoid(probs)

        n_pred_ch = probs.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                probs = torch.softmax(probs, 1)

        if self.other_act is not None:
            probs = self.other_act(probs)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if target.shape != probs.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({probs.shape})")

        B, K, *xyz = probs.shape  # type: ignore

        pc = cast(Tensor, probs[:, self.idc, ...].type(torch.float32))
        tc = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        target_dm_npy: np.ndarray = np.stack([one_hot2hd_dist(tc[b].cpu().detach().numpy())
                                              for b in range(B)], axis=0)

        tdm: Tensor = torch.tensor(target_dm_npy, device=probs.device, dtype=torch.float32)

        pred_segmentation: Tensor = probs2one_hot(probs).cpu().detach()
        pred_dm_npy: np.nparray = np.stack([one_hot2hd_dist(pred_segmentation[b, self.idc, ...].numpy())
                                            for b in range(B)], axis=0)

        pdm: Tensor = torch.tensor(pred_dm_npy, device=probs.device, dtype=torch.float32)

        delta = (pc - tc) ** 2
        dtm = tdm ** self.alpha + pdm ** self.alpha

        # multipled = einsum("bkwh,bkwh->bkwh", delta, dtm)
        hd_loss = einsum("bkxyz,bkxyz->bkxyz", delta, dtm)

        if self.reduction == LossReduction.MEAN.value:
            hd_loss = torch.mean(hd_loss)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            hd_loss = torch.sum(hd_loss)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return hd_loss


class DiceHDLoss(_Loss):
    """
    Compute both Dice loss and Hausdorff Loss, and return the sum of these two losses.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth_nr` and `smooth_dr` parameters are
    values added for dice loss part to the intersection and union components of the inter-over-union calculation
    to smooth results respectively, these values should be small. The `include_background` class attribute can be
    set to False for an instance of the loss to exclude the first category (channel index 0) which is by convention
    assumed to be background. If the non-background segmentations are small compared to the total image size they can get
    overwhelmed by the signal from the background so excluding it in such cases helps convergence.
    """

    def __init__(
            self,
            idc_hd: List,
            alpha_hd: int = 2,
            hybrid: bool = False,
            weight_hd: float = 0.5,
            weight_dice: float = 0.5,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Optional[Callable] = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: str = "mean",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
    ) -> None:
        """
        Args:
            ``ce_weight`` is only used for cross entropy loss, ``reduction`` is used for both losses and other
            parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.hd = HausdorffLoss(
            idc=idc_hd,
            alpha=alpha_hd,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
        )

        self.hybrid = hybrid
        self.weight_hd = weight_hd
        self.weight_dice = weight_dice

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is nither 1 or the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        hd_loss = self.hd(input, target)

        if self.hybrid:
            total_loss: torch.Tensor = self.weight_dice * dice_loss + self.weight_hd * hd_loss
        else:
            total_loss: torch.Tensor = dice_loss + hd_loss

        return total_loss


class SurfaceLoss(_Loss):
    def __init__(self, idc: List,
                 to_onehot_y: bool = False,
                 sigmoid: bool = False,
                 softmax: bool = False,
                 other_act: Optional[Callable] = None,
                 reduction: Union[LossReduction, str] = LossReduction.MEAN,
                 ):
        super().__init__(reduction=LossReduction(reduction).value)
        self.idc = idc
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        print(f"Initialized {self.__class__.__name__} with {self.idc}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:

        if isinstance(dist_maps, Sequence):
            dist_maps = dist_maps[
                1]  # ugly but because input take all loss type and distance map was added at indexing 1

        if self.sigmoid:
            probs = torch.sigmoid(probs)

        n_pred_ch = probs.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                probs = torch.softmax(probs, 1)

        if self.other_act is not None:
            probs = self.other_act(probs)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                dist_maps = one_hot(dist_maps, num_classes=n_pred_ch)

        if dist_maps.shape != probs.shape:
            raise AssertionError(f"distance map has different shape ({dist_maps.shape}) from input ({probs.shape})")

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        loss = einsum("bkxyz,bkxyz->bkxyz", pc, dc)

        if self.reduction == LossReduction.MEAN.value:
            loss = torch.mean(loss)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            loss = torch.sum(loss)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            pass  # returns [N, n_classes] losses
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return loss


BoundaryLoss = SurfaceLoss


class DiceBoundaryLoss(_Loss):
    """
    Compute both Dice loss and Boundary Loss, and return the sum of these two losses.
    Input logits `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).
    Axis N of `input` is expected to have logit predictions for each class rather than being image channels,
    while the same axis of `target` can be 1 or N (one-hot format). The `smooth_nr` and `smooth_dr` parameters are
    values added for dice loss part to the intersection and union components of the inter-over-union calculation
    to smooth results respectively, these values should be small. The `include_background` class attribute can be
    set to False for an instance of the loss to exclude the first category (channel index 0) which is by convention
    assumed to be background. If the non-background segmentations are small compared to the total image size they can get
    overwhelmed by the signal from the background so excluding it in such cases helps convergence.
    """

    def __init__(
            self,
            idc_boundary: List,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Optional[Callable] = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: str = "mean",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            lambda_dice: float = 1.0,
            lambda_boundary: float = 1.0,
    ) -> None:
        """
        Args:
            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example:
                `other_act = torch.tanh`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_boundary: the trade-off weight value for boundary loss. The value should be no less than 0.0.
                Defaults to 1.0.
        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.boundary = BoundaryLoss(
            idc=idc_boundary,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            reduction=reduction,
        )

        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_boundary < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is nither 1 or the same as input.

        """

        dice_loss = self.dice(input, target[0])
        boundary_loss = self.boundary(input, target[1])

        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_boundary * boundary_loss

        return total_loss


class DiceCELoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Optional[Callable] = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: str = "mean",
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            ce_weight: Optional[torch.Tensor] = None,
            lambda_dice: float = 1.0,
            lambda_ce: float = 1.0,
    ) -> None:
        """
        Args:
            ``ce_weight`` and ``lambda_ce`` are only used for cross entropy loss.
            ``reduction`` is used for both losses and other parameters are only used for dice loss.

            include_background: if False channel index 0 (background category) is excluded from the calculation.
            to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            softmax: if True, apply a softmax function to the prediction, only used by the `DiceLoss`,
                don't need to specify activation function for `CrossEntropyLoss`.
            other_act: if don't want to use `sigmoid` or `softmax`, use other callable function to execute
                other activation layers, Defaults to ``None``. for example: `other_act = torch.tanh`.
                only used by the `DiceLoss`, don't need to specify activation function for `CrossEntropyLoss`.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``. The dice loss should
                as least reduce the spatial dimensions, which is different from cross entropy loss, thus here
                the ``none`` option cannot be used.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            ce_weight: a rescaling weight given to each class for cross entropy loss.
                See ``torch.nn.CrossEntropyLoss()`` for more information.
            lambda_dice: the trade-off weight value for dice loss. The value should be no less than 0.0.
                Defaults to 1.0.
            lambda_ce: the trade-off weight value for cross entropy loss. The value should be no less than 0.0.
                Defaults to 1.0.

        """
        super().__init__()
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=ce_weight,
            reduction=reduction,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def ce(self, input: torch.Tensor, target: torch.Tensor):
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]

        target = target.type_as(input)  # just added this line for compatibility
        if n_pred_ch == n_target_ch:
            # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            target = torch.argmax(target, dim=1)
        else:
            target = torch.squeeze(target, dim=1)
        target = target.long()
        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError("the number of dimensions for input and target should be the same.")

        dice_loss = self.dice(input, target)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
