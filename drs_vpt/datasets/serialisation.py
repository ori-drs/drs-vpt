# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
Fit a point cloud's voxel grid size so its integer grid coordinates fit in a fixed number of bits (Sonata's
serialization depth).
"""

import numpy as np


def grid_size_from_depth(points: np.ndarray, max_depth: int) -> float:
    """
    Smallest grid size that guarantees floor((p - min)/gs) <= 2^D - 1, for D=max_depth.
    Uses strict bound: gs > span / 2^D, plus a tiny nextafter bump for safety.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if max_depth <= 0:
        raise ValueError("max_depth must be >= 1")

    pts = points.astype(np.float64)
    mins = pts.min(axis=0)
    span = pts.max(axis=0) - mins
    denom = float(1 << max_depth)  # 2^D

    per_axis_required = np.divide(span, denom, out=np.zeros_like(span), where=(span > 0))
    required = float(per_axis_required.max())

    # strict inequality needs a nudge upward
    return np.nextafter(required, np.inf)


def points_to_grid_fit(points: np.ndarray, grid_size: float, max_depth: int, margin: float = 0.0):
    """
    Adjust grid_size upward so coords fit within max_depth bits.
    Returns (coords, used_grid_size, depth_used).
    """
    min_gs = grid_size_from_depth(points, max_depth)
    used_gs = max(grid_size, min_gs * (1.0 + margin))

    pts = points.astype(np.float64)
    mins = pts.min(axis=0)
    coords = np.floor((pts - mins) / used_gs).astype(np.int64)

    # Final guard: if any spill (shouldn't happen), bump once by 1 ULP and recompute
    M = (1 << max_depth) - 1
    if coords.max() > M:
        used_gs = np.nextafter(used_gs, np.inf)
        coords = np.floor((pts - mins) / used_gs).astype(np.int64)

    depth_used = int((int(coords.max()) + 1).bit_length())
    used_gs = used_gs.astype(np.float32)
    return coords, used_gs, depth_used
