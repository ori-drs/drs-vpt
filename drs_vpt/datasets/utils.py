# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
Point cloud and depth map helpers for the dataset loaders: transforms, depth map back-projection, the per-sample cloud
view and its features, and orientation/voxel utilities.
"""

import numpy as np

from drs_vpt.datasets.serialisation import points_to_grid_fit


def apply_transform_from_left(T, points):
    return T[:3, :3].dot(points.T).T + T[:3, -1]


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def get_lidar_view(view):
    normals = view['normals']
    grid_size = view['grid_size']
    point = dict()
    point["coord"] = view['coord']
    point["color"] = view['color']
    point["normal"] = normals
    point["grid_size"] = np.array([grid_size]).astype(np.float32)
    point["scene_scale"] = np.array([view['scene_scale']]).astype(np.float32)
    return point



def extract_view(T_world_camera, cloud_in_world_frame_xyz, grid_size, max_points=128 * 1024):
    """max_points: random subsample cap before Sonata; 0 or None keeps every point."""
    T_camera_world = np.linalg.inv(T_world_camera)
    cloud_in_world_frame_xyz = apply_transform_from_left(
        T_camera_world, cloud_in_world_frame_xyz[:, :3]
    )
    cloud_in_camera_frame = cloud_in_world_frame_xyz[:, :3]
    scene_scale = np.mean(np.linalg.norm(cloud_in_camera_frame, axis=1))
    scene_scale = np.log1p(scene_scale) + 1e-8
    normals_in_camera_frame = np.zeros_like(cloud_in_world_frame_xyz[:, :3])
    geometry_features_in_camera_frame = np.zeros_like(cloud_in_world_frame_xyz[:, :3])
    # geometry_features_in_camera_frame = compute_geometry_features(cloud_in_camera_frame=cloud_in_camera_frame,
    #                                                               normals_in_camera_frame=normals_in_camera_frame)
    npts = cloud_in_camera_frame.shape[0]
    if max_points and npts > max_points:
        inliers = np.random.choice(npts, max_points, replace=False)
        cloud_in_camera_frame = cloud_in_camera_frame[inliers]
        normals_in_camera_frame = normals_in_camera_frame[inliers]
        geometry_features_in_camera_frame = geometry_features_in_camera_frame[inliers]

    normalized_scene_coord = cloud_in_camera_frame / scene_scale
    normalized_scene_grid_size = grid_size / scene_scale

    lidar_view = dict()
    lidar_view.update(dict(coord=normalized_scene_coord))
    lidar_view.update(dict(normals=normals_in_camera_frame))
    lidar_view.update(dict(color=geometry_features_in_camera_frame))
    lidar_view.update(dict(scene_scale=scene_scale))
    max_depth = 15  # constrain here
    _, grid_size, _ = points_to_grid_fit(normalized_scene_coord,
                                         grid_size=normalized_scene_grid_size,
                                         max_depth=max_depth)
    lidar_view.update(dict(grid_size=grid_size))
    return get_lidar_view(lidar_view)


def compute_geometry_features(cloud_in_camera_frame, normals_in_camera_frame):
    # depth (Euclidean distance from camera)
    depth = np.linalg.norm(cloud_in_camera_frame, axis=1)  # (N,)
    eps = 1e-6
    safe_depth = np.clip(depth, eps, None)  # avoid zero

    # unit ray directions (camera -> point)
    ray_dirs = cloud_in_camera_frame / safe_depth[:, None]  # (N, 3)

    # 1) incidence: | n · (-ray_dir) | in [0, 1]
    cos_incidence = np.abs(
        np.sum(normals_in_camera_frame * (-ray_dirs), axis=1)
    )

    # 2) log depth: log(1 + d), safe at d=0
    log_depth = np.log1p(safe_depth)

    # 3) inverse depth: 1 / d, safe via clipping above
    inv_depth = 1.0 / safe_depth

    geom_feats = np.stack(
        [
            cos_incidence.astype(np.float32),
            log_depth.astype(np.float32),
            inv_depth.astype(np.float32),
        ],
        axis=1,
    )
    return geom_feats


def _rotz90_ccw_4x4() -> np.ndarray:
    """+90° about camera Z (CCW)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return T


def ensure_orientation_from_prescribed_resolution_ccw(
    image_hwc: np.ndarray,  # (H,W,3) uint8/float
    depth_hw: np.ndarray,  # (H,W) float
    K: np.ndarray,  # (3,3) float
    T_w_c: np.ndarray,  # (4,4) camera-to-world (c2w)
    prescribed_resolution_wh: tuple[int, int],  # (W_pres, H_pres)
):
    """
    If the current view orientation (landscape/portrait) doesn't match the prescribed one,
    rotate the raster data 90° CCW, update K via pixel homography, and update c2w via
    right-multiplying a +90° camera-frame rotation about Z.

    Returns: (image2, depth2, K2, T_w_c2, did_rotate)
    """
    H, W = depth_hw.shape[:2]
    W_pres, H_pres = prescribed_resolution_wh

    # Determine whether each is landscape (W>=H) or portrait (H>W)
    cur_landscape = (W >= H)
    pres_landscape = (W_pres >= H_pres)

    need_rotate = (cur_landscape != pres_landscape)
    if not need_rotate:
        return image_hwc, depth_hw, K.astype(np.float32), T_w_c.astype(np.float32), False

    # --- rotate image/depth 90° CCW ---
    image2 = np.rot90(image_hwc, k=1)  # CCW
    depth2 = np.rot90(depth_hw, k=1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # --- intrinsics: K_new = R_img * K ---
    K2 = np.array([
        [fy, 0, cy],
        [0, fx, (W - 1) - cx],
        [0, 0, 1]
    ]).astype(np.float32)

    # --- pose (camera-to-world): T_w_c_new = T_w_c * T_z(+90°) ---
    T90 = _rotz90_ccw_4x4()
    T_w_c2 = (T_w_c @ T90).astype(np.float32)

    return image2, depth2, K2, T_w_c2, True


def voxel_downsample_first(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    points: (N,3) float array
    returns: (M,3) float array, one point per occupied voxel
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    # shift to avoid huge indices if points are far from origin (optional but helps)
    p0 = points.min(axis=0)
    coords = np.floor((points - p0) / voxel_size).astype(np.int64)

    # unique voxel coords; keep first occurrence
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[unique_idx]
