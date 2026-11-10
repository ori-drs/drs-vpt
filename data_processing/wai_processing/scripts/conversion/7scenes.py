# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
- Loads vilens and image data from OSD or GrandTour
- Assumes vilens has run successfully and that the extrinsic and intrinsic calibrations are available
- Assumes there are exact matching timestamps between images and undistorted clouds

"""

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Iterable, Union

import numpy as np
from argconf import argconf_parse
from tqdm import tqdm

from drs_vpt.datasets.utils import depthmap_to_absolute_camera_coordinates

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
)

wai_lock = threading.Lock()

from drs_vpt._vendor.mapanything.utils.wai.core import store_data


def get_7scenes_names(
    cfg,
):
    # Get all scene names to process
    all_scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

    return all_scenes


def process_stamp(seq_path, frame_index, target_scene_root):

    # Undistort and resize image
    undistorted_image = cv2.imread(os.path.join(seq_path, f"{frame_index}.color.png"))
    rescale_factor = 1.0
    undistorted_image = cv2.resize(undistorted_image, None, fx=rescale_factor, fy=rescale_factor,
                                   interpolation=cv2.INTER_LANCZOS4)

    camera_pose = np.loadtxt(os.path.join(seq_path, f"{frame_index}.pose.txt")).astype(np.float64)
    K = np.array(
        [
            [525, 0, 320],
            [0, 525, 240],
            [0, 0, 1.0],
        ]
    ).astype(np.float64)

    # Create depth map
    depth = cv2.imread(
        os.path.join(seq_path, f"{frame_index}.depth.png"), cv2.IMREAD_UNCHANGED) / 1000.0
    depth[depth > 10] = 0.0
    K_depth = np.array(
        [
            [585, 0, 320],
            [0, 585, 240],
            [0, 0, 1.0],
        ]
    ).astype(np.float64)
    Trgb_depth = np.array([[1.0, 0.0, 0.0, -0.0254],
                           [0.0, 1.0, 0.0, -0.00013],
                           [0.0, 0.0, 1.0, -0.00218],
                           [0, 0, 0, 1]])
    cloud_in_camera, mask = depthmap_to_absolute_camera_coordinates(depth,
                                                                         K_depth,
                                                                         camera_pose=Trgb_depth)
    cloud_in_camera = cloud_in_camera[mask]
    new_shape = undistorted_image.shape[:2]
    cloud_pixels = \
        np.round(
            cv2.projectPoints(cloud_in_camera, np.zeros(3), np.zeros(3), K * rescale_factor, np.zeros(4))[
                0].squeeze()).astype(int)
    validity_mask = (cloud_pixels[:, 0] >= 0) & (cloud_pixels[:, 0] < new_shape[1]) & (cloud_pixels[:, 1] >= 0) & (
        cloud_pixels[:, 1] < new_shape[0])
    cloud_pixels = cloud_pixels[validity_mask]
    cloud_in_camera = cloud_in_camera[validity_mask]
    x, y = cloud_pixels.T

    # Create depth map
    depth_image = np.zeros(new_shape, dtype=np.float32)
    depth_image[y, x] = cloud_in_camera[:, 2]
    # Save outputs
    output_prefix = f"{frame_index}"
    rel_target_image_path = Path("images") / f"{output_prefix}.jpg"
    cv2.imwrite(os.path.join(target_scene_root, rel_target_image_path), undistorted_image)
    rel_depth_out_path = Path("depth") / f"{output_prefix}.exr"
    cv2.imwrite(os.path.join(target_scene_root, rel_depth_out_path), depth_image.astype(np.float32))

    new_intrinsics = K * rescale_factor
    new_intrinsics[2, 2] = 1

    # Prepare WAI metadata
    wai_frame = {
        "frame_name": output_prefix,
        "image": str(rel_target_image_path),
        "file_path": str(rel_target_image_path),
        "depth": str(rel_depth_out_path),
        "transform_matrix": camera_pose.tolist(),
        "h": depth.shape[0],
        "w": depth.shape[1],
        "fl_x": float(new_intrinsics[0, 0]),
        "fl_y": float(new_intrinsics[1, 1]),
        "cx": float(new_intrinsics[0, 2]),
        "cy": float(new_intrinsics[1, 2]),
    }

    return wai_frame

def chunked(seq: List[Union[int, str]], chunk_size: int) -> Iterable[List[int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    for i in range(0, len(seq), chunk_size):
        yield seq[i:i + chunk_size]

def slice_camera_files(camera_files: Dict[str, Dict[int, str]], frame_ids: List[int]):
    # camera_files: {cam_name: {frame_id: filepath}}
    frame_set = set(frame_ids)
    return {cam: {fid: fp for fid, fp in files.items() if fid in frame_set}
            for cam, files in camera_files.items()}

def slice_camera_poses(camera_poses: Dict[str, Dict[int, object]], frame_ids: List[int]):
    frame_set = set(frame_ids)
    return {cam: {fid: pose for fid, pose in poses.items() if fid in frame_set}
            for cam, poses in camera_poses.items()}

def slice_lidar_files(lidar_files: Dict[int, str], frame_ids: List[int]):
    frame_set = set(frame_ids)
    return {fid: fp for fid, fp in lidar_files.items() if fid in frame_set}

def slice_lidar_poses(lidar_poses: Dict[int, object], frame_ids: List[int]):
    frame_set = set(frame_ids)
    return {fid: pose for fid, pose in lidar_poses.items() if fid in frame_set}



def run_parallel(seq_path, seq_tsdf_cloud_path, prefixes,target_scene_root):
    wai_frames = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []

        scene_cloud = np.loadtxt(seq_tsdf_cloud_path).astype(np.float64)
        rel_cloud_out_path = Path("map_cloud")  / "map_cloud.npz"
        os.makedirs(os.path.join(target_scene_root, "map_cloud"), exist_ok=True)
        store_data(os.path.join(target_scene_root, rel_cloud_out_path),
                   scene_cloud, "map_cloud")

        for frame_index in prefixes:
            futures.append(executor.submit(
                process_stamp, seq_path, frame_index,
                target_scene_root
            ))

        for future in as_completed(futures):
            wai_frame = future.result()
            with wai_lock:
                if wai_frame is not None:
                    wai_frame["map_cloud"] = str(rel_cloud_out_path)
                    wai_frames.append(wai_frame)

    return wai_frames

def load_nuscenes_camera_files(images_root: str) -> Dict[str, str]:
    folder = images_root
    instance_to_file_path = {x.split(".")[0]: str(os.path.join(folder, x)) for x in os.listdir(folder)
                              if ".png" in x}
    return instance_to_file_path


def load_nuscenes_intrinsics(intrinsics_folder):
    instance_to_file_path = {x.split(".")[0]: str(os.path.join(intrinsics_folder, x)) for x in os.listdir(intrinsics_folder)
                             if ".npy" in x}
    output = dict()
    for k in instance_to_file_path:
        output[k] = np.load(instance_to_file_path[k])
    return output

def load_nuscenes_poses(pose_folder):
    pose_file_names = {x.split(".")[0]: x for x in os.listdir(pose_folder) if ".npy" in x}
    output_pose = dict()
    for k in pose_file_names:
        output_pose[k] = np.load(os.path.join(pose_folder, pose_file_names[k]))
    return output_pose

def load_nuscenes_clouds(cloud_folder):
    pose_file_names = {x.split(".")[0]: os.path.join(cloud_folder, x) for x in os.listdir(cloud_folder) if ".npy" in x}
    return pose_file_names

def parse_train_seq_ids(train_split_path: str) -> list[int]:
    """
    Returns all sequence ids found in TrainSplit.txt (unique, in file order).
    """
    txt = open(train_split_path, "r", encoding="utf-8", errors="ignore").read()
    nums = [int(x) for x in re.findall(r"-?\d+", txt)]
    if not nums:
        raise ValueError(f"No integers found in {train_split_path}")

    seen = set()
    seq_ids = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            seq_ids.append(n)
    return seq_ids


def process_7scenes_scene(cfg, scene_name):
    """

    """
    scene_root = Path(cfg.original_root) / scene_name

    train_split_path = scene_root / "TrainSplit.txt"

    seq_ids = parse_train_seq_ids(train_split_path)

    for seq_id in seq_ids:
        seq_name = f"seq-{seq_id:02d}"
        seq_path = scene_root / seq_name
        seq_tsdf_cloud_path = scene_root / "tsdf_points_txt" / f"{scene_name}_train_tsdf_points.txt"
        if not os.path.isdir(seq_path):
            print(f"[WARN] Missing {seq_name} at {seq_path}, skipping.")
            continue

        prefixes = [
            x.split(".")[0] for x in os.listdir(seq_path) if ".color.png" in x
        ]

        # ---------- chunked processing ----------
        chunk_size = int(getattr(cfg, "chunk_size", 200))  # parameterize this (Hydra/argparse/etc)

        chunks = chunked(prefixes, chunk_size)
        for chunk_idx, prefixes_in_chunk in tqdm(enumerate(chunks), total=(len(prefixes) // chunk_size) + 1):
            subscene_name = f"{scene_name}_{seq_name}_{chunk_idx:04d}"
            target_scene_root = Path(cfg.root) / subscene_name
            image_dir = target_scene_root / "images"
            depth_dir = target_scene_root / "depth"

            image_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            # If run_parallel internally reads lidar poses from disk, ignore lidar_poses_chunk.
            # If it needs poses passed in, add it to the signature and pass it.
            wai_frames_chunk = run_parallel(
                seq_path=seq_path,
                seq_tsdf_cloud_path=seq_tsdf_cloud_path,
                prefixes=prefixes_in_chunk,
                target_scene_root=target_scene_root
            )

            scene_meta = {
                "scene_name": subscene_name,
                "dataset_name": cfg.dataset_name,
                "version": cfg.version,
                "shared_intrinsics": False,
                "camera_model": "PINHOLE",
                "camera_convention": "opencv",
                "scale_type": "colmap",
                "scene_modalities": {},
                "frames": wai_frames_chunk,
                "frame_modalities": {
                    "image": {"frame_key": "image", "format": "image"},
                    "depth": {"frame_key": "depth", "format": "depth"},
                    "map_cloud": {"frame_key": "map_cloud", "format": "map_cloud"}
                },
            }

            store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    CONVERSION_PATH = Path(__file__).parent
    cfg = argconf_parse(CONVERSION_PATH / "7scenes_config.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_7scenes_scene, cfg, get_original_scene_names_func=get_7scenes_names)
