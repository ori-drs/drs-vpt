# Vendored from PixLoc (https://github.com/cvg/pixloc, commit 6f7a943), pixloc/pixlib/geometry/costs.py.
# Copyright (c) the PixLoc authors (Sarlin et al., CVPR 2021). Apache License 2.0, see LICENSE in this folder.
# Modified for DRS VPT (Oxford Robotics Institute): normalized-feature Jacobian (compute_normalized_J_f_p2D*), image/map confidence weighting.
# The exact differences to upstream are in pixloc.patch.

import torch
from typing import Optional, Tuple
from torch import Tensor

from .wrappers import Pose, Camera
from .optimization import J_normalization
from .interpolation import Interpolator


class DirectAbsoluteCost:
    def __init__(self, interpolator: Interpolator, normalize: bool = False):
        self.interpolator = interpolator
        self.normalize = normalize

    def residuals(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None,
            do_gradients: bool = False):

        p3D_q = T_w2q * p3D
        p2D, visible = camera.world2image(p3D_q)

        F_p2D_raw, valid, gradients = self.interpolator(
            F_query, p2D, return_gradients=do_gradients)
        valid = valid & visible

        if confidences is not None:
            C_ref, C_query = confidences
            C_query_p2D, _, _ = self.interpolator(
                C_query, p2D, return_gradients=False)
            weight = C_ref * C_query_p2D
            weight = weight.squeeze(-1).masked_fill(~valid, 0.)
        else:
            weight = None

        if self.normalize:
            F_p2D = torch.nn.functional.normalize(F_p2D_raw, dim=-1)
        else:
            F_p2D = F_p2D_raw

        res = F_p2D - F_ref
        info = (p3D_q, F_p2D_raw, gradients)
        return res, valid, weight, F_p2D, info

    def jacobian(
            self, T_w2q: Pose, camera: Camera,
            p3D_q: Tensor, F_p2D_raw: Tensor, J_f_p2D_raw: Tensor):

        J_p3D_T = T_w2q.J_transform(p3D_q)
        J_p2D_p3D, _ = camera.J_world2image(p3D_q)

        if self.normalize:
            # J_f_p2D = self.compute_normalized_J_f_p2D(F_p2D_raw, J_f_p2D_raw)
            J_f_p2D = self.compute_normalized_J_f_p2D_efficient(F_p2D_raw, J_f_p2D_raw)
        else:
            J_f_p2D = J_f_p2D_raw

        J_p2D_T = J_p2D_p3D @ J_p3D_T
        J = J_f_p2D @ J_p2D_T
        return J, J_p2D_T

    def compute_normalized_J_f_p2D(self, F_p2D_raw, J_f_p2D_raw):
        return J_normalization(F_p2D_raw) @ J_f_p2D_raw

    def compute_normalized_J_f_p2D_efficient(self, F_p2D_raw, J_f_p2D_raw):
        eps = 1e-6
        x = F_p2D_raw
        x_normed = torch.nn.functional.normalize(x, dim=-1)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        small = (norm.squeeze(-1) < eps)
        norm = norm.clamp(min=eps)
        left = J_f_p2D_raw / norm.unsqueeze(-1)
        right = (x_normed.unsqueeze(-1) @ (x_normed.unsqueeze(-2) @ J_f_p2D_raw)) / norm.unsqueeze(-1)
        """
            The operation is J_normalization @ J_f_p2D_raw
                        (I / x_norm -  x_normalized @ x_normalized^T) J_f_p2D_raw
            To avoid materializing the big normalization matrix, do:
            J_f_p2D_raw / x_norm - x_normalized * ( x_normalized^T J_f_p2D_raw)
            We call this left - right
        """
        result = torch.where(small[..., None, None], J_f_p2D_raw, left - right)
        return result

    def residual_jacobian(
            self, T_w2q: Pose, camera: Camera, p3D: Tensor,
            F_ref: Tensor, F_query: Tensor,
            confidences: Optional[Tuple[Tensor, Tensor]] = None):

        res, valid, weight, F_p2D, info = self.residuals(
            T_w2q, camera, p3D, F_ref, F_query, confidences, True)
        J, _ = self.jacobian(T_w2q, camera, *info)
        J = J.masked_fill(~valid[..., None, None], 0.0)
        return res, valid, weight, F_p2D, J


