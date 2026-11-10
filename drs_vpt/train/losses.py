# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), mapanything/train/losses.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.
# Modified for DRS VPT (Oxford Robotics Institute): FactoredGeometryScaleRegr3D also supervises the point cloud branch
# (cloud point map, cloud pose before and after alignment) and the reprojection error at each coarse-to-fine level.
# Only the losses DRS VPT trains with are kept.

"""
Multi-view geometric losses for training DRS VPT.
References: DUSt3R & MASt3R
"""

import math
from copy import copy, deepcopy

import torch
import torch.nn as nn

from drs_vpt._vendor.mapanything.utils.geometry import (
    angle_diff_vec3,
    apply_log_to_norm,
    closed_form_pose_inverse,
    geotrf,
    normalize_multiple_pointclouds,
    rotation_matrix_to_quaternion,
    transform_pose_using_quats_and_trans_2_to_1,
)
from drs_vpt._vendor.pixloc.wrappers import Pose


def compute_reprojection_error(gt_projections, pr_projections):
    reprojection_valid_mask = (gt_projections[1].float() * pr_projections[1].float())[..., None]
    reprojection_errors = torch.linalg.norm(
        (gt_projections[0] - pr_projections[0]) * reprojection_valid_mask,
        dim=-1, keepdim=True)
    # Ensure mask is float for multiplication
    mask = reprojection_valid_mask.float()
    # Apply mask
    masked_errors = reprojection_errors * mask  # (B, N, 1)
    # Sum errors and count valid points per batch
    sum_errors = masked_errors.sum(dim=1)  # (B, 1)
    num_valid = mask.sum(dim=1)  # (B, 1)
    # Safe mean: avoid division by zero
    safe_num_valid = num_valid.clamp_min(1.0)
    mean_error = sum_errors / safe_num_valid
    mean_error = torch.where(num_valid > 0, mean_error, torch.zeros_like(mean_error))
    return mean_error


def get_loss_terms_and_details(losses_dict, valid_masks, self_name, n_views):
    """
    Helper function to generate loss terms and details for different loss types.

    Args:
        losses_dict (dict): Dictionary mapping loss types to their values.
            Format: {
                'loss_type': {
                    'values': list_of_loss_tensors or single_tensor,
                    'use_mask': bool,
                    'is_multi_view': bool
                }
            }
        valid_masks (list): List of valid masks for each view, flattened like the losses (B, H * W).
        self_name (str): Name of the loss class.
        n_views (int): Number of views.

    Returns:
        tuple: (loss_terms, details) where loss_terms is a list of tuples (loss, mask, type)
               and details is a dictionary of loss details.
    """
    loss_terms = []
    details = {}

    for loss_type, loss_info in losses_dict.items():
        values = loss_info["values"]
        use_mask = loss_info["use_mask"]
        is_multi_view = loss_info["is_multi_view"]
        if is_multi_view:
            # Handle multi-view losses (list of tensors)
            view_loss_details = []
            for i in range(n_views):
                mask = valid_masks[i] if use_mask else None
                loss_terms.append((values[i], mask, loss_type))

                # Add details for individual view
                values_after_masking = values[i][mask] if use_mask else values[i]

                if values_after_masking.numel() > 0:
                    view_loss_detail = float(values_after_masking.mean())
                    if view_loss_detail > 0:
                        details[f"{self_name}_{loss_type}_view{i + 1}"] = (
                            view_loss_detail
                        )
                        view_loss_details.append(view_loss_detail)
            # Add average across views
            if len(view_loss_details) > 0:
                details[f"{self_name}_{loss_type}_avg"] = sum(view_loss_details) / len(
                    view_loss_details
                )
        else:
            # Handle single tensor losses
            if values is not None:
                loss_terms.append((values, None, loss_type))
                if values.numel() > 0:
                    loss_detail = float(values.mean())
                    if loss_detail > 0:
                        details[f"{self_name}_{loss_type}"] = loss_detail

    return loss_terms, details


def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)


def compute_normal_loss(points, gt_points, mask):
    """
    Compute the normal loss between the predicted and ground truth points.
    References:
    https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/moge/train/losses.py#L205

    Args:
        points (torch.Tensor): Predicted points. Shape: (..., H, W, 3).
        gt_points (torch.Tensor): Ground truth points. Shape: (..., H, W, 3).
        mask (torch.Tensor): Mask indicating valid points. Shape: (..., H, W).

    Returns:
        torch.Tensor: Normal loss.
    """
    height, width = points.shape[-3:-1]

    leftup, rightup, leftdown, rightdown = (
        points[..., :-1, :-1, :],
        points[..., :-1, 1:, :],
        points[..., 1:, :-1, :],
        points[..., 1:, 1:, :],
    )
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = (
        gt_points[..., :-1, :-1, :],
        gt_points[..., :-1, 1:, :],
        gt_points[..., 1:, :-1, :],
        gt_points[..., 1:, 1:, :],
    )
    gt_upxleft = torch.cross(
        gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1
    )
    gt_leftxdown = torch.cross(
        gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1
    )
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(
        gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1
    )

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = (
        mask[..., :-1, :-1],
        mask[..., :-1, 1:],
        mask[..., 1:, :-1],
        mask[..., 1:, 1:],
    )
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = (
        mask_upxleft
        * _smooth(
        angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE),
        beta=BETA_RAD,
    )
        + mask_leftxdown
        * _smooth(
        angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE),
        beta=BETA_RAD,
    )
        + mask_downxright
        * _smooth(
        angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE),
        beta=BETA_RAD,
    )
        + mask_rightxup
        * _smooth(
        angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE),
        beta=BETA_RAD,
    )
    )

    total_valid_mask = mask_upxleft | mask_leftxdown | mask_downxright | mask_rightxup
    valid_count = total_valid_mask.sum()
    if valid_count > 0:
        loss = loss.sum() / (valid_count * (4 * max(points.shape[-3:-1])))
    else:
        loss = 0 * loss.sum()

    return loss


def compute_gradient_loss(prediction, gt_target, mask):
    """
    Compute the gradient loss between the prediction and GT target at valid points.
    References:
    https://docs.nerf.studio/_modules/nerfstudio/model_components/losses.html#GradientLoss
    https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py

    Args:
        prediction (torch.Tensor): Predicted scene representation. Shape: (B, H, W, C).
        gt_target (torch.Tensor): Ground truth scene representation. Shape: (B, H, W, C).
        mask (torch.Tensor): Mask indicating valid points. Shape: (B, H, W).
    """
    # Expand mask to match number of channels in prediction
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    summed_mask = torch.sum(mask, (1, 2, 3))

    # Compute the gradient of the prediction and GT target
    diff = prediction - gt_target
    diff = torch.mul(mask, diff)

    # Gradient in x direction
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Gradient in y direction
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp the outlier gradients
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Compute the total loss
    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    num_valid_pixels = torch.sum(summed_mask)
    if num_valid_pixels > 0:
        image_loss = torch.sum(image_loss) / num_valid_pixels
    else:
        image_loss = 0 * torch.sum(image_loss)

    return image_loss


def compute_gradient_matching_loss(prediction, gt_target, mask, scales=4):
    """
    Compute the multi-scale gradient matching loss between the prediction and GT target at valid points.
    This loss biases discontinuities to be sharp and to coincide with discontinuities in the ground truth.
    More info in MiDAS: https://arxiv.org/pdf/1907.01341.pdf; Equation 11
    References:
    https://docs.nerf.studio/_modules/nerfstudio/model_components/losses.html#GradientLoss
    https://github.com/autonomousvision/monosdf/blob/main/code/model/loss.py

    Args:
        prediction (torch.Tensor): Predicted scene representation. Shape: (B, H, W, C).
        gt_target (torch.Tensor): Ground truth scene representation. Shape: (B, H, W, C).
        mask (torch.Tensor): Mask indicating valid points. Shape: (B, H, W).
        scales (int): Number of scales to compute the loss at. Default: 4.
    """
    # Define total loss
    total_loss = 0.0

    # Compute the gradient loss at different scales
    for scale in range(scales):
        step = pow(2, scale)
        grad_loss = compute_gradient_loss(
            prediction[:, ::step, ::step],
            gt_target[:, ::step, ::step],
            mask[:, ::step, ::step],
        )
        total_loss += grad_loss

    return total_loss


def Sum(*losses_and_masks):
    """
    Aggregates multiple losses into a single loss value or returns the original losses.

    Args:
        *losses_and_masks: Variable number of tuples, each containing (loss, mask, rep_type)
            - loss: Tensor containing loss values
            - mask: Mask indicating valid pixels/regions
            - rep_type: String indicating the type of representation (e.g., 'pts3d', 'depth')

    Returns:
        If the first loss has dimensions > 0:
            Returns the original list of (loss, mask, rep_type) tuples
        Otherwise:
            Returns a scalar tensor that is the sum of all loss values
    """
    loss, mask, rep_type = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2, rep_type2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    "Base Criterion to support different reduction methods"

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    "L-norm loss"

    def forward(self, a, b, **kwargs):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 4, (
            f"Bad shape = {a.shape}"
        )
        dist = self.distance(a, b, **kwargs)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b, **kwargs):
        raise NotImplementedError()


class L1Loss(LLoss):
    "L1 distance"

    def distance(self, a, b, **kwargs):
        return torch.abs(a - b).sum(dim=-1)


class L2Loss(LLoss):
    "Euclidean (L2 Norm) distance"

    def distance(self, a, b, **kwargs):
        return torch.norm(a - b, dim=-1)


class RobustRegressionLoss(LLoss):
    """
    Generalized Robust Loss introduced in https://arxiv.org/abs/1701.03077.
    """

    def __init__(self, alpha=0.5, scaling_c=0.25, reduction="mean"):
        """
        Initialize the Robust Regression Loss.

        Args:
            alpha (float): Shape parameter controlling the robustness of the loss.
                Lower values make the loss more robust to outliers. Default: 0.5.
            scaling_c (float): Scale parameter controlling the transition between
                quadratic and robust behavior. Default: 0.1.
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__(reduction)
        self.alpha = alpha
        self.scaling_c = scaling_c

    def distance(self, a, b, **kwargs):
        error_scaled = torch.sum(((a - b) / self.scaling_c) ** 2, dim=-1)
        robust_loss = (abs(self.alpha - 2) / self.alpha) * (
            torch.pow((error_scaled / abs(self.alpha - 2)) + 1, self.alpha / 2) - 1
        )
        return robust_loss


class BCELoss(BaseCriterion):
    """Binary Cross Entropy loss"""

    def forward(self, predicted_logits, reference_mask):
        """
        Args:
            predicted_logits: (B, H, W) tensor of predicted logits for the mask
            reference_mask: (B, H, W) tensor of reference mask

        Returns:
            loss: scalar tensor of the BCE loss
        """
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            predicted_logits, reference_mask.float()
        )

        return bce_loss


class Criterion(nn.Module):
    """
    Base class for all criterion modules that wrap a BaseCriterion.

    This class serves as a wrapper around BaseCriterion objects, providing
    additional functionality like naming and reduction mode control.

    Args:
        criterion (BaseCriterion): The base criterion to wrap.
    """

    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, BaseCriterion), (
            f"{criterion} is not a proper criterion!"
        )
        self.criterion = copy(criterion)

    def get_name(self):
        """
        Returns a string representation of this criterion.

        Returns:
            str: A string containing the class name and the wrapped criterion.
        """
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        """
        Creates a deep copy of this criterion with the specified reduction mode.

        This method recursively sets the reduction mode for this criterion and
        any chained MultiLoss criteria.

        Args:
            mode (str): The reduction mode to set. Default: "none".

        Returns:
            Criterion: A new criterion with the specified reduction mode.
        """
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """
    Base class for combinable loss functions with automatic tracking of individual loss values.

    This class enables easy combination of multiple loss functions through arithmetic operations:
        loss = MyLoss1() + 0.1*MyLoss2()

    The combined loss functions maintain their individual weights and the forward pass
    automatically computes and aggregates all losses while tracking individual loss values.

    Usage:
        Inherit from this class and override get_name() and compute_loss() methods.

    Attributes:
        _alpha (float): Weight multiplier for this loss component.
        _loss2 (MultiLoss): Reference to the next loss in the chain, if any.
    """

    def __init__(self):
        """Initialize the MultiLoss with default weight of 1 and no chained loss."""
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        """
        Compute the loss value for this specific loss component.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor or tuple: Either the loss tensor or a tuple of (loss, details_dict).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def get_name(self):
        """
        Get the name of this loss component.

        Returns:
            str: The name of the loss.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def __mul__(self, alpha):
        """
        Multiply the loss by a scalar weight.

        Args:
            alpha (int or float): The weight to multiply the loss by.

        Returns:
            MultiLoss: A new loss object with the updated weight.

        Raises:
            AssertionError: If alpha is not a number.
        """
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # Support both loss*alpha and alpha*loss

    def __add__(self, loss2):
        """
        Add another loss to this loss, creating a chain of losses.

        Args:
            loss2 (MultiLoss): Another loss to add to this one.

        Returns:
            MultiLoss: A new loss object representing the combined losses.

        Raises:
            AssertionError: If loss2 is not a MultiLoss.
        """
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # Find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        """
        Create a string representation of the loss, including weights and chained losses.

        Returns:
            str: String representation of the loss.
        """
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        """
        Compute the weighted loss and aggregate with any chained losses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The total weighted loss.
                - dict: Details about individual loss components.
        """
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class NonAmbiguousMaskLoss(Criterion, MultiLoss):
    """
    Loss on non-ambiguous mask prediction logits.
    """

    def __init__(self, criterion):
        super().__init__(criterion)

    def compute_loss(self, batch, preds, **kw):
        """
        Args:
            batch: {"image_views": list of dicts with the gt data, "cloud": the cloud input}
            preds: {"image_views": list of dicts with the predictions, "cloud": the cloud outputs}

        Returns:
            loss: Sum class of the lossses for N-views and the loss details
        """
        # Init loss list to keep track of individual losses for each view
        loss_list = []
        mask_loss_details = {}
        mask_loss_total = 0
        self_name = type(self).__name__

        # Loop over the views
        for view_idx, (gt, pred) in enumerate(zip(batch["image_views"], preds["image_views"])):
            # Get the GT non-ambiguous masks
            gt_non_ambiguous_mask = gt["non_ambiguous_mask"]

            # Get the predicted non-ambiguous mask logits
            pred_non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]

            # Compute the loss for the current view
            loss = self.criterion(pred_non_ambiguous_mask_logits, gt_non_ambiguous_mask)

            # Add the loss to the list
            loss_list.append((loss, None, "non_ambiguous_mask"))

            # Add the loss details to the dictionary
            mask_loss_details[f"{self_name}_mask_view{view_idx + 1}"] = float(loss)
            mask_loss_total += float(loss)

        # Compute the average loss across all views
        mask_loss_details[f"{self_name}_mask_avg"] = mask_loss_total / len(batch["image_views"])

        return Sum(*loss_list), (mask_loss_details | {})


def _select_loss_sets(losses, set_indices, n_views):
    """
    The per-view loss terms of the selected loss sets, and their indices in losses.
    Each loss set is one rep_type across all views, so set idx covers losses[idx * n_views:(idx + 1) * n_views].
    """
    selected_losses = []
    selected_indices = set()
    for idx in set_indices:
        start_idx = idx * n_views
        end_idx = min((idx + 1) * n_views, len(losses))
        selected_losses.extend(losses[start_idx:end_idx])
        selected_indices.update(range(start_idx, end_idx))
    return selected_losses, selected_indices


def _keep_bottom_n_percent(tensor, mask, bottom_n_percent):
    """
    Keep the bottom n percent of each instance's valid per-pixel loss values.

    Args:
        tensor (torch.Tensor): The tensor containing the per-pixel loss values.
                               Shape: (B, N) where B is the batch size and N is the number of total pixels.
        mask (torch.Tensor): The mask indicating valid pixels. Shape: (B, N).
        bottom_n_percent (float): The percentage of valid values to keep. Range: [0, 100].

    Returns:
        torch.Tensor: Flattened tensor containing the bottom n percent of per-pixel loss values.
    """
    B, N = tensor.shape

    # Calculate the number of valid elements (where mask is True)
    num_valid = mask.sum(dim=1)

    # Calculate the number of elements to keep (n% of valid elements)
    num_keep = (num_valid * bottom_n_percent / 100).long()

    # Create a mask for the bottom n% elements
    keep_mask = torch.arange(N, device=tensor.device).unsqueeze(
        0
    ) < num_keep.unsqueeze(1)

    # Create a tensor with inf where mask is False
    masked_tensor = torch.where(
        mask, tensor, torch.tensor(float("inf"), device=tensor.device)
    )

    # Sort the masked tensor along the N dimension
    sorted_tensor, _ = torch.sort(masked_tensor, dim=1, descending=False)

    # Get the bottom n% elements
    bottom_n_percent_elements = sorted_tensor[keep_mask]

    return bottom_n_percent_elements


def _exclude_top_n_percent(loss, msk, is_synthetic, bottom_n_percent, apply_to_real_data_only):
    """
    The valid per-pixel loss values of one view, flattened, without the top N% of each real-data instance.

    Args:
        loss (torch.Tensor): Per-pixel loss values. Shape: (B, N), or (1, N) for a cloud loss.
        msk (torch.Tensor or None): Valid pixels. Shape: (B, N); None if all are valid.
        is_synthetic (torch.Tensor): Whether each instance is synthetic data. Shape: (B,).
        bottom_n_percent (float): The percentage of valid values to keep. Range: [0, 100].
        apply_to_real_data_only (bool): Keep every valid value of synthetic data instances.
    """
    if msk is None:
        msk = torch.ones_like(loss).bool()

    # A cloud loss holds the valid points of the whole batch packed into one row, so it is not split by instance
    if loss.shape[0] == 1 and is_synthetic.shape[0] != 1:
        return _keep_bottom_n_percent(loss, msk, bottom_n_percent)

    if apply_to_real_data_only:
        synthetic_mask = is_synthetic
        real_data_mask = ~is_synthetic
    else:
        synthetic_mask = torch.zeros_like(is_synthetic)
        real_data_mask = torch.ones_like(is_synthetic)

    aggregated_losses = []
    if synthetic_mask.any():
        aggregated_losses.append(loss[synthetic_mask][msk[synthetic_mask]])
    if real_data_mask.any():
        aggregated_losses.append(
            _keep_bottom_n_percent(loss[real_data_mask], msk[real_data_mask], bottom_n_percent)
        )
    return torch.cat(aggregated_losses, dim=0)


def _add_view_detail(loss_details, valid_views, key_prefix, view_idx, value):
    """Record a view's loss value as <key_prefix>_view<n> and update the running average <key_prefix>_avg."""
    loss_details[f"{key_prefix}_view{view_idx + 1}"] = value
    avg_key = f"{key_prefix}_avg"
    if avg_key not in loss_details:
        loss_details[avg_key] = value
        valid_views[avg_key] = 1
    else:
        valid_views[avg_key] += 1
        loss_details[avg_key] += (value - loss_details[avg_key]) / valid_views[avg_key]


def _add_remaining_losses(total_loss, losses, processed_indices):
    """Add the plain masked mean of every loss term not in processed_indices to total_loss."""
    for idx, (loss, msk, rep_type) in enumerate(losses):
        if idx in processed_indices:
            continue
        loss_after_masking = loss[msk] if msk is not None else loss
        loss_mean = loss_after_masking.mean() if loss_after_masking.numel() > 0 else 0
        total_loss = total_loss + loss_mean
    return total_loss


class ExcludeTopNPercentPixelLoss(MultiLoss):
    """
    Pixel-level regression loss where for each instance in a batch the top N% of per-pixel loss values are ignored
    for the mean loss computation.
    Allows selecting which pixel-level regression loss sets to apply the exclusion to.
    """

    def __init__(
        self,
        pixel_loss,
        top_n_percent=5,
        apply_to_real_data_only=True,
        loss_set_indices=None,
    ):
        """
        Args:
            pixel_loss (MultiLoss): The pixel-level regression loss to be used.
            top_n_percent (float): The percentage of top per-pixel loss values to ignore. Range: [0, 100]. Default: 5.
            apply_to_real_data_only (bool): Whether to apply the loss only to real world data. Default: True.
            loss_set_indices (list or None): Indices of the loss sets to apply the exclusion to.
                Each index selects a specific loss set across all views (with the same rep_type).
                If None, defaults to [0] which applies to the first loss set only.
        """
        super().__init__()
        self.pixel_loss = pixel_loss.with_reduction("none")
        self.top_n_percent = top_n_percent
        self.bottom_n_percent = 100 - top_n_percent
        self.apply_to_real_data_only = apply_to_real_data_only
        self.loss_set_indices = [0] if loss_set_indices is None else loss_set_indices

    def get_name(self):
        return f"ExcludeTopNPercentPixelLoss({self.pixel_loss})"

    def compute_loss(self, batch, preds, **kw):
        # Compute per-pixel loss
        losses, details = self.pixel_loss(batch, preds, **kw)
        n_views = len(batch["image_views"])
        selected_losses, processed_indices = _select_loss_sets(losses, self.loss_set_indices, n_views)

        total_loss = 0.0
        loss_details = {}
        valid_views = {}
        self_name = type(self.pixel_loss).__name__

        # Process selected losses with top N percent exclusion
        for loss_idx, (loss, msk, rep_type) in enumerate(selected_losses):
            view_idx = loss_idx % n_views  # Map to corresponding view index
            if loss.numel() == 0:
                continue

            view_loss = _exclude_top_n_percent(
                loss, msk, batch["image_views"][view_idx]["is_synthetic"], self.bottom_n_percent,
                self.apply_to_real_data_only,
            )
            if view_loss.numel() > 0:
                view_loss = view_loss.mean()
                total_loss = total_loss + view_loss
                _add_view_detail(loss_details, valid_views, f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_loss",
                                 view_idx, float(view_loss))

        # Add unmodified losses for sets not in selected_losses
        total_loss = _add_remaining_losses(total_loss, losses, processed_indices)

        return total_loss, dict(**loss_details, **details)


class ConfAndExcludeTopNPercentPixelLoss(MultiLoss):
    """
    Combined loss that applies ConfLoss to one set of pixel-level regression losses
    and ExcludeTopNPercentPixelLoss to another set of pixel-level regression losses.
    """

    def __init__(
        self,
        pixel_loss,
        conf_alpha=1,
        top_n_percent=5,
        apply_to_real_data_only=True,
        conf_loss_set_indices=None,
        exclude_loss_set_indices=None,
    ):
        """
        Args:
        pixel_loss (MultiLoss): The pixel-level regression loss to be used.
        conf_alpha (float): Alpha parameter for ConfLoss. Default: 1.
        top_n_percent (float): Percentage of top per-pixel loss values to ignore. Range: [0, 100]. Default: 5.
        apply_to_real_data_only (bool): Whether to apply the exclude loss only to real world data. Default: True.
        conf_loss_set_indices (list or None): Indices of the loss sets to apply confidence weighting to.
            Each index selects a specific loss set across all views (with the same rep_type).
            If None, defaults to [0] which applies to the first loss set only.
        exclude_loss_set_indices (list or None): Indices of the loss sets to apply top N percent exclusion to.
            Each index selects a specific loss set across all views (with the same rep_type).
            If None, defaults to [1] which applies to the second loss set only.
        """
        super().__init__()
        self.pixel_loss = pixel_loss.with_reduction("none")
        assert conf_alpha > 0
        self.conf_alpha = conf_alpha
        self.top_n_percent = top_n_percent
        self.bottom_n_percent = 100 - top_n_percent
        self.apply_to_real_data_only = apply_to_real_data_only
        self.conf_loss_set_indices = (
            [0] if conf_loss_set_indices is None else conf_loss_set_indices
        )
        self.exclude_loss_set_indices = (
            [1] if exclude_loss_set_indices is None else exclude_loss_set_indices
        )

    def get_name(self):
        return f"ConfAndExcludeTopNPercentPixelLoss({self.pixel_loss})"

    def compute_loss(self, batch, preds, **kw):
        # Compute per-pixel loss
        losses, pixel_loss_details = self.pixel_loss(batch, preds, **kw)
        n_views = len(batch["image_views"])
        conf_selected_losses, conf_processed_indices = _select_loss_sets(losses, self.conf_loss_set_indices, n_views)
        exclude_selected_losses, exclude_processed_indices = _select_loss_sets(
            losses, self.exclude_loss_set_indices, n_views
        )

        # Initialize total loss and details
        total_loss = 0
        loss_details = {}
        valid_views = {}
        self_name = type(self.pixel_loss).__name__

        # Process selected losses with confidence weighting
        for loss_idx, (loss, msk, rep_type) in enumerate(conf_selected_losses):
            view_idx = loss_idx % n_views  # Map to corresponding view index
            if loss.numel() == 0:
                continue

            # Get the confidence, flattened like the per-pixel loss
            if "cloud_" in rep_type:
                assert msk is None  # The cloud losses cover every point
                conf = preds["cloud"]["cloud_conf"]
                conf = conf.view(conf.shape[0], -1)
            else:
                conf = preds["image_views"][view_idx]["conf"]
                conf = conf.view(conf.shape[0], -1)[msk]
                loss = loss[msk]

            # Weight the loss by the confidence
            conf_loss = loss * conf - self.conf_alpha * torch.log(conf)

            # Only add to total loss and store details if there are valid elements
            if conf_loss.numel() > 0:
                conf_loss = conf_loss.mean()
                total_loss = total_loss + conf_loss
                _add_view_detail(loss_details, valid_views, f"{self_name}_{rep_type}_conf_loss", view_idx,
                                 float(conf_loss))

        # Process selected losses with top N percent exclusion
        for loss_idx, (loss, msk, rep_type) in enumerate(exclude_selected_losses):
            view_idx = loss_idx % n_views  # Map to corresponding view index
            if loss.numel() == 0:
                continue

            view_loss = _exclude_top_n_percent(
                loss, msk, batch["image_views"][view_idx]["is_synthetic"], self.bottom_n_percent,
                self.apply_to_real_data_only,
            )
            if view_loss.numel() > 0:
                view_loss = view_loss.mean()
                total_loss = total_loss + view_loss
                _add_view_detail(loss_details, valid_views, f"{self_name}_{rep_type}_bot{self.bottom_n_percent}%_loss",
                                 view_idx, float(view_loss))

        # Add unmodified losses for sets not processed with either confidence or exclusion
        total_loss = _add_remaining_losses(
            total_loss, losses, conf_processed_indices.union(exclude_processed_indices)
        )

        return total_loss, dict(**loss_details, **pixel_loss_details)


def _flatten_hw(x):
    """(B, H, W, C) -> (B, H * W, C)"""
    return x.view(x.shape[0], -1, x.shape[-1])


def _loss_set(values, use_mask=False, is_multi_view=True):
    return {"values": values, "use_mask": use_mask, "is_multi_view": is_multi_view}


class FactoredGeometryScaleRegr3DPlusNormalGMLoss(Criterion, MultiLoss):
    """
    Regression, Normals & Gradient Matching Loss for Factored Geometry & Scale, and for the point cloud branch.
    """

    def __init__(
        self,
        criterion,
        norm_mode="avg_dis",
        cam_frame_points_loss_weight=1,
        depth_loss_weight=1,
        ray_directions_loss_weight=1,
        pose_quats_loss_weight=1,
        pose_trans_loss_weight=1,
        scale_loss_weight=1,
        world_frame_points_loss_weight=1,
        apply_normal_and_gm_loss_to_synthetic_data_only=True,
        normal_loss_weight=1,
        gm_loss_weight=1,
    ):
        """
        Initialize the loss criterion for Factored Geometry (Ray Directions, Depth, Pose), Scale
        and the Collective Geometry i.e. Local Frame Pointmaps & World Frame Pointmaps.
        Additionally computes:
        (1) Normal Loss over the Camera Frame Pointmaps in euclidean coordinates,
        (2) Gradient Matching (GM) Loss over the Depth Z in log space. (MiDAS applied GM loss in disparity space)
        (3) The point cloud branch losses: cloud point map, cloud pose before and after alignment, and the
            reprojection error at each coarse-to-fine alignment level.
        The predicted and ground truth scene representations are normalized w.r.t. the frame of view0 and compared
        per pixel (H x W flattened) in log space. Loss is applied between the predicted metric scale and the ground
        truth metric scale.

        Args:
            criterion (BaseCriterion): The base criterion to use for computing the loss.
            norm_mode (str): Normalization mode for the gt and predicted scene representation. Default: "avg_dis".
            cam_frame_points_loss_weight (float): Weight to use for the camera frame pointmap loss. Default: 1.
            depth_loss_weight (float): Weight to use for the depth loss. Default: 1.
            ray_directions_loss_weight (float): Weight to use for the ray directions loss. Default: 1.
            pose_quats_loss_weight (float): Weight to use for the pose quats losses. Default: 1.
            pose_trans_loss_weight (float): Weight to use for the pose trans and reprojection losses. Default: 1.
            scale_loss_weight (float): Weight to use for the scale loss. Default: 1.
            world_frame_points_loss_weight (float): Weight to use for the world frame and cloud pointmap losses.
                Default: 1.
            apply_normal_and_gm_loss_to_synthetic_data_only (bool): If True, apply the normal and gm loss only to synthetic data.
                If False, apply the normal and gm loss to all data. Default: True.
            normal_loss_weight (float): Weight to use for the normal loss. Default: 1.
            gm_loss_weight (float): Weight to use for the gm loss. Default: 1.
        """
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.cam_frame_points_loss_weight = cam_frame_points_loss_weight
        self.depth_loss_weight = depth_loss_weight
        self.ray_directions_loss_weight = ray_directions_loss_weight
        self.pose_quats_loss_weight = pose_quats_loss_weight
        self.pose_trans_loss_weight = pose_trans_loss_weight
        self.scale_loss_weight = scale_loss_weight
        self.world_frame_points_loss_weight = world_frame_points_loss_weight
        self.apply_normal_and_gm_loss_to_synthetic_data_only = (
            apply_normal_and_gm_loss_to_synthetic_data_only
        )
        self.normal_loss_weight = normal_loss_weight
        self.gm_loss_weight = gm_loss_weight
        self.reprojection_criterion = L1Loss()
        self.scale_criterion = L2Loss()

    def get_all_info(self, batch, preds):
        """
        Function to get all the information needed to compute the loss.
        Returns all quantities normalized w.r.t. camera of view0.
        """
        n_views = len(batch["image_views"])

        # Everything is normalized w.r.t. camera of view0
        in_camera0 = closed_form_pose_inverse(batch["image_views"][0]["camera_pose"])
        # Ground truth quantities
        no_norm_gt_pts = []
        no_norm_gt_pts_cam = []
        no_norm_gt_depth = []
        no_norm_gt_pose_trans = []
        valid_masks = []
        gt_ray_directions = []
        gt_pose_quats = []
        # Predicted quantities
        no_norm_pr_pts = []
        no_norm_pr_pts_cam = []
        no_norm_pr_depth = []
        no_norm_pr_pose_trans = []
        pr_ray_directions = []
        pr_pose_quats = []
        metric_pr_pts_to_compute_scale = []

        # Get ground truth & prediction info for all views
        for i in range(n_views):
            view = batch["image_views"][i]
            pred = preds["image_views"][i]

            # Get the ground truth
            no_norm_gt_pts.append(geotrf(in_camera0, view["pts3d"]))
            valid_masks.append(view["valid_mask"].clone())
            no_norm_gt_pts_cam.append(view["pts3d_cam"])
            gt_ray_directions.append(view["ray_directions_cam"])
            no_norm_gt_depth.append(view["depth_along_ray"])
            if i == 0:
                # For view0, initialize identity pose
                batch_size = gt_ray_directions[0].shape[0]
                tensor_args = dict(dtype=gt_ray_directions[0].dtype, device=gt_ray_directions[0].device)
                gt_pose_quats.append(torch.tensor([0, 0, 0, 1], **tensor_args).unsqueeze(0).repeat(batch_size, 1))
                no_norm_gt_pose_trans.append(torch.tensor([0, 0, 0], **tensor_args).unsqueeze(0).repeat(batch_size, 1))
            else:
                # For other views, transform pose to view0's frame
                gt_pose_quats_in_view0, no_norm_gt_pose_trans_in_view0 = transform_pose_using_quats_and_trans_2_to_1(
                    batch["image_views"][0]["camera_pose_quats"],
                    batch["image_views"][0]["camera_pose_trans"],
                    view["camera_pose_quats"],
                    view["camera_pose_trans"],
                )
                gt_pose_quats.append(gt_pose_quats_in_view0)
                no_norm_gt_pose_trans.append(no_norm_gt_pose_trans_in_view0)

            # The predictions are in view0's frame. Divide by the predicted metric scaling factor to get the raw
            # predicted points, depth_along_ray, and pose_trans; this detaches the predicted metric scaling factor from
            # the geometry based loss.
            metric_scaling_factor = pred["metric_scaling_factor"]
            curr_view_no_norm_pr_pts = pred["pts3d"] / metric_scaling_factor.unsqueeze(-1).unsqueeze(-1)
            no_norm_pr_pts.append(curr_view_no_norm_pr_pts)
            no_norm_pr_pts_cam.append(pred["pts3d_cam"] / metric_scaling_factor.unsqueeze(-1).unsqueeze(-1))
            no_norm_pr_depth.append(pred["depth_along_ray"] / metric_scaling_factor.unsqueeze(-1).unsqueeze(-1))
            no_norm_pr_pose_trans.append(pred["cam_trans"] / metric_scaling_factor)
            pr_ray_directions.append(pred["ray_directions"])
            pr_pose_quats.append(pred["cam_quats"])

            # Get the predicted metric scale points. Detach the raw predicted points so that the scale loss is only
            # applied to the scaling factor
            metric_pr_pts_to_compute_scale.append(
                curr_view_no_norm_pr_pts.detach() * metric_scaling_factor.unsqueeze(-1).unsqueeze(-1)
            )

        # The cloud: the ground truth points and scanner pose in view0's frame, and the predictions divided by view0's
        # predicted metric scaling factor
        T_world_scanner = batch["cloud"]["scanner_pose"]
        no_norm_gt_cloud_pts = geotrf(in_camera0, geotrf(T_world_scanner, preds["cloud"]["resampled_cloud_coords"]))
        gt_cloud_pose_quats, no_norm_gt_cloud_pose_trans = transform_pose_using_quats_and_trans_2_to_1(
            batch["image_views"][0]["camera_pose_quats"],
            batch["image_views"][0]["camera_pose_trans"],
            rotation_matrix_to_quaternion(T_world_scanner[:, :3, :3]),
            T_world_scanner[:, :3, -1],
        )
        view0_metric_scaling_factor = preds["image_views"][0]["metric_scaling_factor"]
        no_norm_pr_cloud_pts = preds["cloud"]["cloud_pts3d"] / view0_metric_scaling_factor.unsqueeze(-1)
        no_norm_pr_cloud_pose_trans = preds["cloud"]["cloud_trans"] / view0_metric_scaling_factor
        no_norm_pr_cloud_pose_opt_trans = preds["cloud"]["cloud_trans_opt"] / view0_metric_scaling_factor[..., None]

        # Normalize the predicted and ground truth points
        pr_normalization_output = normalize_multiple_pointclouds(
            no_norm_pr_pts, valid_masks, self.norm_mode, ret_factor=True
        )
        pr_pts = pr_normalization_output[:-1]
        pr_norm_factor = pr_normalization_output[-1]
        gt_normalization_output = normalize_multiple_pointclouds(
            no_norm_gt_pts, valid_masks, self.norm_mode, ret_factor=True
        )
        gt_pts = gt_normalization_output[:-1]
        gt_norm_factor = gt_normalization_output[-1]

        pr_pts_cam = [pts_cam / pr_norm_factor for pts_cam in no_norm_pr_pts_cam]
        pr_depth = [depth / pr_norm_factor for depth in no_norm_pr_depth]
        pr_pose_trans = [trans / pr_norm_factor[:, :, 0, 0] for trans in no_norm_pr_pose_trans]
        gt_pts_cam = [pts_cam / gt_norm_factor for pts_cam in no_norm_gt_pts_cam]
        gt_depth = [depth / gt_norm_factor for depth in no_norm_gt_depth]
        gt_pose_trans = [trans / gt_norm_factor[:, :, 0, 0] for trans in no_norm_gt_pose_trans]

        pr_cloud_pts = no_norm_pr_cloud_pts / pr_norm_factor[:, :, 0]
        pr_cloud_pose_trans = no_norm_pr_cloud_pose_trans / pr_norm_factor[:, :, 0, 0]
        pr_cloud_pose_opt_trans = no_norm_pr_cloud_pose_opt_trans / pr_norm_factor[:, :, :, 0]
        gt_cloud_pose_trans = no_norm_gt_cloud_pose_trans / gt_norm_factor[:, :, 0, 0]
        gt_cloud_pts = no_norm_gt_cloud_pts / gt_norm_factor[:, :, 0]

        # Get the mask indicating ground truth metric scale quantities
        metric_scale_mask = batch["image_views"][0]["is_metric_scale"]
        valid_gt_norm_factor_mask = (
            gt_norm_factor[:, 0, 0, 0] > 1e-8
        )  # Mask out cases where depth for all views is invalid
        valid_metric_scale_mask = metric_scale_mask & valid_gt_norm_factor_mask

        if valid_metric_scale_mask.any():
            # Compute the scale norm factor using the predicted metric scale points
            metric_pr_normalization_output = normalize_multiple_pointclouds(
                metric_pr_pts_to_compute_scale,
                valid_masks,
                self.norm_mode,
                ret_factor=True,
            )
            pr_metric_norm_factor = metric_pr_normalization_output[-1]

            # Get the valid ground truth and predicted scale norm factors for the metric ground truth quantities
            gt_metric_norm_factor = gt_norm_factor[valid_metric_scale_mask]
            pr_metric_norm_factor = pr_metric_norm_factor[valid_metric_scale_mask]
        else:
            gt_metric_norm_factor = None
            pr_metric_norm_factor = None

        # Reprojection error of the cloud points at each alignment level, ground truth pose vs aligned pose
        gt_T_cam0_scanner = torch.linalg.inv(batch["image_views"][0]["camera_pose"]) @ batch["cloud"]["scanner_pose"]
        gt_T_cam0_scanner = Pose.from_Rt(gt_T_cam0_scanner[..., :3, :3],
                                         gt_T_cam0_scanner[..., :3, -1])
        gt_reprojections_at_level = [
            camera_0.world2image(gt_T_cam0_scanner * preds["cloud"]["scene_coords_for_reprojection"])
            for camera_0 in preds["cloud"]["cameras_at_level"]
        ]
        pr_reprojections_at_level = preds["cloud"]["reprojections_at_level"]

        reprojection_losses = [
            compute_reprojection_error(gt, pr).clip(max=5.0)
            for gt, pr in zip(gt_reprojections_at_level, pr_reprojections_at_level)
        ]

        # Pack the valid cloud points of the whole batch into one row: (1, N, 3)
        valid_cloud_mask = preds["cloud"]["valid_cloud_mask"]
        gt_cloud_pts = gt_cloud_pts[valid_cloud_mask][None]
        pr_cloud_pts = pr_cloud_pts[valid_cloud_mask][None]

        # Pack into info dicts
        gt_info = []
        pred_info = []
        for i in range(n_views):
            gt_info.append(
                {
                    "ray_directions": gt_ray_directions[i],
                    "depth_along_ray": gt_depth[i],
                    "pose_trans": gt_pose_trans[i],
                    "pose_quats": gt_pose_quats[i],
                    "pts3d": gt_pts[i],
                    "pts3d_cam": gt_pts_cam[i],
                }
            )
            pred_info.append(
                {
                    "ray_directions": pr_ray_directions[i],
                    "depth_along_ray": pr_depth[i],
                    "pose_trans": pr_pose_trans[i],
                    "pose_quats": pr_pose_quats[i],
                    "pts3d": pr_pts[i],
                    "pts3d_cam": pr_pts_cam[i],
                }
            )
        gt_cloud_info = {
            "pts3d": gt_cloud_pts,
            "pose_trans": gt_cloud_pose_trans,
            "pose_quats": gt_cloud_pose_quats,
        }
        pred_cloud_info = {
            "pts3d": pr_cloud_pts,
            "pose_trans": pr_cloud_pose_trans,
            "pose_quats": preds["cloud"]["cloud_quats"],
            "pose_opt_trans": pr_cloud_pose_opt_trans,
            "pose_opt_quats": preds["cloud"]["cloud_quats_opt"],
        }

        return (
            gt_info,
            pred_info,
            gt_cloud_info,
            pred_cloud_info,
            valid_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
            reprojection_losses,
        )

    def _quats_loss(self, pred_quats, gt_quats, factor):
        # Handle quaternion two-to-one mapping
        return torch.minimum(
            self.criterion(pred_quats, gt_quats, factor=factor),
            self.criterion(pred_quats, -gt_quats, factor=factor),
        )

    def compute_loss(self, batch, preds, **kw):
        (
            gt_info,
            pred_info,
            gt_cloud_info,
            pred_cloud_info,
            valid_masks,
            gt_metric_norm_factor,
            pr_metric_norm_factor,
            reprojection_losses,
        ) = self.get_all_info(batch, preds)
        n_views = len(batch["image_views"])

        # Mask out samples in the batch where the gt depth validity mask is entirely zero
        valid_norm_factor_masks = [
            mask.sum(dim=(1, 2)) > 0 for mask in valid_masks
        ]  # List of (B,)

        normal_losses = []
        gradient_matching_losses = []
        pose_trans_losses = []
        pose_quats_losses = []
        ray_directions_losses = []
        depth_losses = []
        cam_pts3d_losses = []
        pts3d_losses = []

        for i in range(n_views):
            gt = gt_info[i]
            pred = pred_info[i]

            # Get the camera frame points, log space depth_z & valid masks
            pred_depth_z = apply_log_to_norm(pred["pts3d_cam"][..., 2:])
            gt_depth_z = apply_log_to_norm(gt["pts3d_cam"][..., 2:])
            valid_mask_for_normal_gm_loss = valid_masks[i]

            # Update the validity mask for normal & gm loss based on the synthetic data mask if required
            if self.apply_normal_and_gm_loss_to_synthetic_data_only:
                synthetic_mask = batch["image_views"][i]["is_synthetic"]  # (B, )
                synthetic_mask = synthetic_mask.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                synthetic_mask = synthetic_mask.expand(
                    -1, pred_depth_z.shape[1], pred_depth_z.shape[2]
                )  # (B, H, W)
                valid_mask_for_normal_gm_loss = (
                    valid_mask_for_normal_gm_loss & synthetic_mask
                )

            # Compute the normal loss
            normal_loss = compute_normal_loss(pred["pts3d_cam"], gt["pts3d_cam"], valid_mask_for_normal_gm_loss)
            normal_losses.append(normal_loss * self.normal_loss_weight)

            # Compute the gradient matching loss
            gradient_matching_loss = compute_gradient_matching_loss(
                pred_depth_z, gt_depth_z, valid_mask_for_normal_gm_loss
            )
            gradient_matching_losses.append(gradient_matching_loss * self.gm_loss_weight)

            # Flatten the H x W dimensions to H*W; depth and points are compared in log space
            gt_ray_directions = _flatten_hw(gt["ray_directions"])
            pred_ray_directions = _flatten_hw(pred["ray_directions"])
            gt_depth = apply_log_to_norm(_flatten_hw(gt["depth_along_ray"]))
            pred_depth = apply_log_to_norm(_flatten_hw(pred["depth_along_ray"]))
            gt_cam_pts3d = apply_log_to_norm(_flatten_hw(gt["pts3d_cam"]))
            pred_cam_pts3d = apply_log_to_norm(_flatten_hw(pred["pts3d_cam"]))
            gt_pts3d = apply_log_to_norm(_flatten_hw(gt["pts3d"]))
            pred_pts3d = apply_log_to_norm(_flatten_hw(pred["pts3d"]))
            valid_masks[i] = valid_masks[i].view(valid_masks[i].shape[0], -1)

            # Compute pose translation and rotation losses
            pred_pose_trans = pred["pose_trans"][valid_norm_factor_masks[i]]
            gt_pose_trans = gt["pose_trans"][valid_norm_factor_masks[i]]
            pose_trans_loss = self.criterion(pred_pose_trans, gt_pose_trans, factor="pose_trans")
            pose_trans_losses.append(pose_trans_loss * self.pose_trans_loss_weight)
            pose_quats_loss = self._quats_loss(pred["pose_quats"], gt["pose_quats"], "pose_quats")
            pose_quats_losses.append(pose_quats_loss * self.pose_quats_loss_weight)

            # Compute ray direction loss
            ray_directions_loss = self.criterion(pred_ray_directions, gt_ray_directions, factor="ray_directions")
            ray_directions_losses.append(ray_directions_loss * self.ray_directions_loss_weight)

            # Compute depth loss
            depth_loss = self.criterion(pred_depth, gt_depth, factor="depth")
            depth_losses.append(depth_loss * self.depth_loss_weight)

            # Compute camera frame point loss
            cam_pts3d_loss = self.criterion(pred_cam_pts3d, gt_cam_pts3d, factor="points")
            cam_pts3d_losses.append(cam_pts3d_loss * self.cam_frame_points_loss_weight)

            # Compute world frame point loss
            pts3d_loss = self.criterion(pred_pts3d, gt_pts3d, factor="points")
            pts3d_losses.append(pts3d_loss * self.world_frame_points_loss_weight)

        # The cloud losses do not depend on the view; like the per-view losses, they are counted once per view
        gt_cloud_pose_trans = gt_cloud_info["pose_trans"][valid_norm_factor_masks[0]]
        gt_cloud_pose_quats = gt_cloud_info["pose_quats"]
        n_opt_levels = pred_cloud_info["pose_opt_trans"].shape[1]

        cloud_pts3d_loss = self.criterion(
            apply_log_to_norm(pred_cloud_info["pts3d"]), apply_log_to_norm(gt_cloud_info["pts3d"]),
            factor="cloud_points",
        )
        cloud_pts3d_loss = cloud_pts3d_loss * self.world_frame_points_loss_weight

        cloud_pose_trans_loss = self.criterion(
            pred_cloud_info["pose_trans"][valid_norm_factor_masks[0]], gt_cloud_pose_trans, factor="cloud_pose_trans"
        )
        cloud_pose_trans_loss = cloud_pose_trans_loss * self.pose_trans_loss_weight

        cloud_pose_quats_loss = self._quats_loss(pred_cloud_info["pose_quats"], gt_cloud_pose_quats, "cloud_pose_quats")
        cloud_pose_quats_loss = cloud_pose_quats_loss * self.pose_quats_loss_weight

        # The losses on the aligned poses (one per alignment level) are logged, not trained on
        cloud_pose_opt_trans_loss = self.criterion(
            pred_cloud_info["pose_opt_trans"][valid_norm_factor_masks[0]],
            gt_cloud_pose_trans.unsqueeze(1).expand(-1, n_opt_levels, -1),
            factor="cloud_pose_opt_trans",
        )
        cloud_pose_opt_trans_loss = (cloud_pose_opt_trans_loss * self.pose_trans_loss_weight).detach()

        cloud_pose_opt_quats_loss = self._quats_loss(
            pred_cloud_info["pose_opt_quats"], gt_cloud_pose_quats.unsqueeze(1).expand(-1, n_opt_levels, -1),
            "cloud_pose_opt_quats",
        )
        cloud_pose_opt_quats_loss = (cloud_pose_opt_quats_loss * self.pose_quats_loss_weight).detach()

        # Compute the scale loss
        if gt_metric_norm_factor is not None:
            scale_loss = (
                self.scale_criterion(
                    apply_log_to_norm(pr_metric_norm_factor), apply_log_to_norm(gt_metric_norm_factor), factor="scale"
                )
                * self.scale_loss_weight
            )
        else:
            scale_loss = None

        # The loss sets, in the order that the loss set indices of the wrapping losses refer to
        losses_dict = {
            "pts3d": _loss_set(pts3d_losses, use_mask=True),
            "cam_pts3d": _loss_set(cam_pts3d_losses, use_mask=True),
            "depth_along_ray": _loss_set(depth_losses, use_mask=True),
            # Cloud input losses
            "cloud_pts3d": _loss_set([cloud_pts3d_loss] * n_views),
            # Image losses
            "ray_directions": _loss_set(ray_directions_losses),
            "pose_quats": _loss_set(pose_quats_losses),
            "pose_trans": _loss_set(pose_trans_losses),
            # This next entry produces one loss for all views not one loss per view. TLDR; it breaks some assumptions
            # downstream in the include exclude conf vs 95 percent logic. No worries, all cloud losses are placed
            # before this.
            "scale": _loss_set(scale_loss, is_multi_view=False),
            "normal": _loss_set(normal_losses),
            "gradient_matching": _loss_set(gradient_matching_losses),
            "cloud_pose_quats": _loss_set([cloud_pose_quats_loss] * n_views),
            "cloud_pose_trans": _loss_set([cloud_pose_trans_loss] * n_views),
        }
        for opt_level in range(n_opt_levels):
            losses_dict[f"cloud_pose_opt{opt_level}_quats"] = _loss_set(
                [cloud_pose_opt_quats_loss[:, opt_level]] * n_views
            )
            losses_dict[f"cloud_pose_opt{opt_level}_trans"] = _loss_set(
                [cloud_pose_opt_trans_loss[:, opt_level]] * n_views
            )
            reprojection_loss_at_level = self.reprojection_criterion(
                reprojection_losses[opt_level],
                torch.zeros_like(reprojection_losses[opt_level])) * self.pose_trans_loss_weight
            losses_dict[f"cloud_pose_opt{opt_level}_pixel_error"] = _loss_set(
                reprojection_loss_at_level, is_multi_view=False
            )
        loss_terms, details = get_loss_terms_and_details(
            losses_dict,
            valid_masks,
            type(self).__name__,
            n_views,
        )
        losses = Sum(*loss_terms)

        return losses, details
