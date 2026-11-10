# Vendored from PixLoc (https://github.com/cvg/pixloc, commit 6f7a943), pixloc/pixlib/models/learned_optimizer.py (+ base_optimizer.py).
# Copyright (c) the PixLoc authors (Sarlin et al., CVPR 2021). Apache License 2.0, see LICENSE in this folder.
# Modified for DRS VPT (Oxford Robotics Institute): rewritten LM loop (run_optimization) with feature normalization and confidence weights.
# The exact differences to upstream are in pixloc.patch.

"""
Implements a simple differentiable optimizer based on Levenberg-Marquardt
with a constant, scalar damping factor and a fixed number of iterations.
"""

import logging
from typing import Tuple, Optional
import torch
from torch import Tensor, nn

from drs_vpt._vendor.pixloc.costs import DirectAbsoluteCost
from drs_vpt._vendor.pixloc.wrappers import Camera, Pose
from drs_vpt._vendor.pixloc.interpolation import Interpolator
from drs_vpt._vendor.pixloc.losses import scaled_barron
from drs_vpt._vendor.pixloc.optimization import optimizer_step

logger = logging.getLogger(__name__)


class LearnedOptimizer(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.normalize_features = True
        self.num_iterations = 5
        self.interpolator = Interpolator()
        self.cost_fn = DirectAbsoluteCost(self.interpolator,
                                          normalize=self.normalize_features)
        self.loss_fn = scaled_barron(0, 0.1)
        self.logging_fn = None

    def run_optimization(self, p3D: Tensor, F_ref: Tensor, F_query: Tensor,
             T_init: Pose, camera: Camera, mask: Optional[Tensor] = None,
             W_ref_query: Optional[Tuple[Tensor, Tensor]] = None):
        assert torch.backends.cuda.matmul.allow_tf32 == True
        assert torch.backends.cudnn.allow_tf32 == True

        T = T_init.detach()
        if self.normalize_features:
            F_ref = torch.nn.functional.normalize(F_ref, dim=-1)
        args = (camera, p3D, F_ref, F_query, W_ref_query)
        failed = torch.full(T.shape, False, dtype=torch.bool, device=T.device)

        for i in range(self.num_iterations):
            res, valid, w_unc, _, J = self.cost_fn.residual_jacobian(T, *args)
            if mask is not None:
                valid &= mask
            failed = failed | (valid.long().sum(-1) < 100)  # too few points

            # compute the cost and aggregate the weights
            cost = (res**2).sum(-1)
            cost, w_loss, _ = self.loss_fn(cost)
            weights = w_loss * valid.float()
            if w_unc is not None:
                weights *= w_unc

            # solve the linear system
            g, H = self.build_system(J, res, weights)
            delta = optimizer_step(g, H, 0.01, mask=~failed)

            # compute the pose update
            dt, dw = delta.split([3, 3], dim=-1)
            T_delta = Pose.from_aa(dw, dt)
            T = T_delta @ T

            self.log(i=i, T_init=T_init, T=T, T_delta=T_delta, cost=cost,
                     valid=valid, w_unc=w_unc, w_loss=w_loss, H=H, J=J)

        if failed.any():
            logger.debug('One batch element had too few valid points.')

        return T, failed

    def build_system(self, J: Tensor, res: Tensor, weights: Tensor):
        w_sum = weights.sum(dim=-1).clamp_min(1.0)  # (...,)
        grad = torch.einsum('...ndi,...nd->...ni', J, res)   # ... x N x 6
        grad = weights[..., None] * grad
        grad = grad.sum(-2) / w_sum[..., None]   # ... x 6

        Hess = torch.einsum('...ijk,...ijl->...ikl', J, J)  # ... x N x 6 x 6
        Hess = weights[..., None, None] * Hess
        Hess = Hess.sum(-3) / w_sum[..., None, None]   # ... x 6 x6

        return grad, Hess

    def log(self, **args):
        if self.logging_fn is not None:
            self.logging_fn(**args)


