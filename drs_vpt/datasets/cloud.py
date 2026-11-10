# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""The point cloud input of a training sample.

Each dataset accumulates the points of several frames into one world-frame cloud (the map the images are localized
in) and turns it into the model's cloud input with make_cloud_input: the cloud is expressed in a scanner frame placed
at a random rotation and offset from one of the source cameras, then cut down to the Sonata input.
Which frames build the cloud and the random choices made while building it are set per dataset by CloudConfig,
from the dataset config (configs/dataset/drs_vpt.yaml, <dataset>.cloud).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from drs_vpt.datasets.utils import extract_view


@dataclass
class CloudConfig:
    """How a sample's cloud is built (configs/dataset/drs_vpt.yaml, <dataset>.cloud)."""

    frames: str = "covisible"
    """Which frames the cloud is accumulated from: "covisible" samples them together with the image views from the
    scene's covisibility graph, disjoint from the image views; "scene_stride" takes every frame_stride-th frame of the
    whole scene."""
    frames_per_image: int = 1
    """Cloud frames sampled per image view. The image views are the first of these covisible frames in both modes;
    with scene_stride the rest are drawn but not used."""
    include_first_image_frame: bool = False
    """covisible: also accumulate the first image view's own frame."""
    frame_stride: int = 10
    """scene_stride: the stride through the scene's frames (sorted by name)."""

    grid_size: List[float] = field(default_factory=lambda: [0.005, 0.05])
    """Range (m) of Sonata's grid size, drawn per sample. MegaDepth scales its SfM point spacing by this range."""
    voxel_size: Optional[List[float]] = None
    """Range (m) of the voxel size the accumulated cloud is downsampled with, or None to keep every point."""
    scanner_offset: float = 10.0
    """Standard deviation (m) of the scanner position around its source camera (or the cloud centroid). MegaDepth: in
    grid-size units."""
    scanner_at_centroid: bool = False
    """Place the scanner around the cloud's centroid instead of around the source camera (small indoor scenes)."""
    lidar_probability: float = 0.5
    """For datasets with both, the chance a source frame contributes its LiDAR scan rather than its depth map."""
    max_range: float = 80.0
    """LiDAR points further than this (m) are dropped."""
    max_points: int = 38 * 1024
    """Random subsample cap on the cloud before Sonata."""

    @classmethod
    def from_config(cls, config):
        return config if isinstance(config, cls) else cls(**(config or {}))


def make_cloud_input(cloud_in_world: np.ndarray, source_camera_pose: np.ndarray, grid_size, scanner_offset: float,
                     max_points: int, at_centroid: bool = False) -> dict:
    """The cloud as model input, in a scanner frame at a random rotation from source_camera_pose (camera-to-world) and
    a Gaussian offset of std scanner_offset from that camera (or, with at_centroid, from the cloud's centroid).
    Returns the Sonata input fields and scanner_pose (scanner-to-world)."""
    T_camera_scanner = np.eye(4, dtype=np.float32)
    T_camera_scanner[:3, :3] = Rotation.random().as_matrix().astype(np.float32)
    if at_centroid:
        scanner_pose = source_camera_pose @ T_camera_scanner
        offset = np.random.randn(1, 3).astype(np.float32) * scanner_offset
        scanner_pose[:3, 3] = cloud_in_world.astype(np.float64).mean(0).astype(np.float32) + offset
    else:
        T_camera_scanner[:3, 3] = (np.random.randn(1, 3).astype(np.float32) * scanner_offset)[0]
        scanner_pose = source_camera_pose @ T_camera_scanner
    cloud = extract_view(scanner_pose, cloud_in_world, grid_size, max_points=max_points)
    cloud["scanner_pose"] = scanner_pose
    return cloud
