# Vendored from PixLoc (https://github.com/cvg/pixloc, commit 6f7a943), pixloc/pixlib/geometry/utils.py.
# Copyright (c) the PixLoc authors (Sarlin et al., CVPR 2021). Apache License 2.0, see LICENSE in this folder.
# Modified for DRS VPT (Oxford Robotics Institute): trimmed to undistort_points / J_undistort_points.
# The exact differences to upstream are in pixloc.patch.

import torch


@torch.jit.script
def undistort_points(pts, dist):
    '''Undistort normalized 2D coordinates
       and check for validity of the distortion model.
    '''
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts ** 2, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2 ** 2
        undist = undist + pts * radial

        # The distortion model is supposedly only valid within the image
        # boundaries. Because of the negative radial distortion, points that
        # are far outside of the boundaries might actually be mapped back
        # within the image. To account for this, we discard points that are
        # beyond the inflection point of the distortion model,
        # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
        limited = ((k2 > 0) & ((9 * k1 ** 2 - 20 * k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(torch.where(
            k2 > 0, (torch.sqrt(9 * k1 ** 2 - 20 * k2) - 3 * k1) / (10 * k2), 1 / (3 * k1)))
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2 * p12 * uv + p21 * (r2 + 2 * pts ** 2)
            # TODO: handle tangential boundaries

    return undist, valid


@torch.jit.script
def J_undistort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]

    J_diag = torch.ones_like(pts)
    J_cross = torch.zeros_like(pts)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts ** 2, -1, keepdim=True)
        uv = torch.prod(pts, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2 ** 2
        d_radial = (2 * k1 + 4 * k2 * r2)
        J_diag += radial + (pts ** 2) * d_radial
        J_cross += uv * d_radial

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            J_diag += 2 * p12 * pts.flip(-1) + 6 * p21 * pts
            J_cross += 2 * p12 * pts + 2 * p21 * pts.flip(-1)

    J = torch.diag_embed(J_diag) + torch.diag_embed(J_cross).flip(-1)
    return J
