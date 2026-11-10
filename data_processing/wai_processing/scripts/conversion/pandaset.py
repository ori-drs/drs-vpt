# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
- Loads vilens and image data from OSD or GrandTour
- Assumes vilens has run successfully and that the extrinsic and intrinsic calibrations are available
- Assumes there are exact matching timestamps between images and undistorted clouds

"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Iterable

import numpy as np
from argconf import argconf_parse
from tqdm import tqdm

from wai_processing.scripts.conversion.pandaset_utils import load_pandaset_camera_files, \
    load_pandaset_json_poses, load_pandaset_json_intrinsics, read_pandaset_cloud
from wai_processing.utils.lidar import warp_cloud_es_ee_three

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
)

wai_lock = threading.Lock()

from drs_vpt._vendor.mapanything.utils.wai.core import store_data


def get_grand_tour_scene_names(
    cfg,
):
    # Get all scene names to process
    scene_names = [x for x in sorted(os.listdir(cfg.original_root))
                   if os.path.isdir(os.path.join(cfg.original_root, x))]
    return scene_names


def process_stamp(cam_name, frame_index, T_world_camera, K, image_path, cloud_path, target_scene_root):
    T_camera_lidar = np.linalg.inv(T_world_camera)

    # Undistort and resize image
    undistorted_image = cv2.imread(image_path)
    rescale_factor = 1.0
    undistorted_image = cv2.resize(undistorted_image, None, fx=rescale_factor, fy=rescale_factor,
                                   interpolation=cv2.INTER_LANCZOS4)
    new_shape = undistorted_image.shape[:2]

    # Process point cloud
    cloud_in_camera_frame = warp_cloud_es_ee_three(T_camera_lidar, read_pandaset_cloud(cloud_path))
    cloud_in_camera = cloud_in_camera_frame[cloud_in_camera_frame[:, 2] > 0.0]
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
    if depth_image.sum() < 50_000:
        return None

    # Save outputs
    output_prefix = f"{frame_index:02d}_{cam_name}"
    rel_target_image_path = Path("images") / f"{output_prefix}.jpg"
    cv2.imwrite(os.path.join(target_scene_root, rel_target_image_path), undistorted_image)
    rel_depth_out_path = Path("depth") / f"{output_prefix}.exr"
    cv2.imwrite(os.path.join(target_scene_root, rel_depth_out_path), depth_image)
    rel_cloud_out_path = Path("cloud") / f"{output_prefix}.npz"
    os.makedirs(os.path.join(target_scene_root, "cloud"), exist_ok=True)
    store_data(os.path.join(target_scene_root, rel_cloud_out_path),
               cloud_in_camera_frame, "cloud")
    cv2.imwrite(os.path.join(target_scene_root, rel_depth_out_path), depth_image)

    new_intrinsics = K * rescale_factor
    new_intrinsics[2, 2] = 1

    # Prepare WAI metadata
    wai_frame = {
        "frame_name": output_prefix,
        "image": str(rel_target_image_path),
        "file_path": str(rel_target_image_path),
        "depth": str(rel_depth_out_path),
        "cloud": str(rel_cloud_out_path),
        "transform_matrix": T_world_camera.tolist(),
        "h": depth_image.shape[0],
        "w": depth_image.shape[1],
        "fl_x": float(new_intrinsics[0, 0]),
        "fl_y": float(new_intrinsics[1, 1]),
        "cx": float(new_intrinsics[0, 2]),
        "cy": float(new_intrinsics[1, 2]),
    }

    return wai_frame

def chunked(seq: List[int], chunk_size: int) -> Iterable[List[int]]:
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



def run_parallel(camera_files, lidar_files, camera_intrinsics, camera_poses, target_scene_root,
                 indices):
    wai_frames = []
    with tqdm(total=len(lidar_files) * len(camera_files), desc="Processing Stamps") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for cam_name in camera_files:
                for frame_index in indices:
                    futures.append(executor.submit(
                        process_stamp, cam_name, frame_index,
                        camera_poses[cam_name][frame_index],
                        camera_intrinsics[cam_name],
                        camera_files[cam_name][frame_index],
                        lidar_files[frame_index],
                        target_scene_root
                    ))

            for future in as_completed(futures):
                wai_frame = future.result()
                with wai_lock:
                    if wai_frame is not None:
                        wai_frames.append(wai_frame)
                pbar.update(1)

    return wai_frames


def process_pandaset_scene(cfg, scene_name):
    """
    Process a BlendedMVS scene into the WAI format.
    The PFM format depth maps are converted to default WAI depth format (exr).

    Expected input structure (OSD):
        scene-level
        ├── images
        │        ├── cam0
        │            ├──  1710252325.945682506.jpg
        │            ├──  1710252326.945682506.jpg
        │            ├──  ...
        │        ├── cam1
        │            ├──  1710252325.945682506.jpg
        │        └── cam2
        │            ├──  1710252325.945682506.jpg
        ├── slam-poses.csv
        ├── sensor.yaml
        └── undist-clouds
            ├── cloud_1710252304_395841875.pcd
            ├── cloud_1710252313_395773906.pcd
            ├── cloud_1710252314_195768106.pcd
            ├── cloud_1710252314_695764506.pcd
            ├── cloud_1710252315_195760806.pcd
            ├── cloud_1710252316_557081000.pcd

    """
    scene_root = Path(cfg.original_root) / scene_name

    images_folder = os.path.join(scene_root, "camera")
    camera_files = load_pandaset_camera_files(images_folder)
    camera_ids = {k: list(v.keys()) for k, v in camera_files.items()}
    camera_poses = dict()
    for k in camera_ids:
        camera_poses_path = os.path.join(scene_root, f"camera/{k}/poses.json")
        camera_poses[k] = load_pandaset_json_poses(camera_poses_path)

    camera_intrinsics = dict()
    for k in camera_ids:
        camera_intrinsics_path = os.path.join(scene_root, f"camera/{k}/intrinsics.json")
        camera_intrinsics[k] = load_pandaset_json_intrinsics(camera_intrinsics_path)

    for k in camera_ids:
        assert (len(camera_poses[k]) == len(camera_files[k]))

    undistorted_clouds_folder = os.path.join(scene_root, "lidar")
    lidar_files = {int(x.split(".")[0]): os.path.join(undistorted_clouds_folder, x)
                   for x in os.listdir(undistorted_clouds_folder) if "pkl" in x}
    lidar_ids = list(lidar_files.keys())

    lidar_poses_path = os.path.join(scene_root, "lidar/poses.json")
    lidar_poses = load_pandaset_json_poses(lidar_poses_path)
    assert (len(lidar_poses) == len(lidar_ids))

    for k in camera_ids:
        assert set(camera_ids[k]) == set(lidar_ids)

    # ---------- chunked processing ----------
    chunk_size = int(getattr(cfg, "chunk_size", 50))  # parameterize this (Hydra/argparse/etc)

    # choose a deterministic order
    common_ids = sorted(lidar_ids)

    for chunk_idx, ids_chunk in enumerate(chunked(common_ids, chunk_size)):
        subscene_name = f"{scene_name}_{chunk_idx:04d}"
        target_scene_root = Path(cfg.root) / subscene_name
        image_dir = target_scene_root / "images"
        depth_dir = target_scene_root / "depth"

        image_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # If run_parallel internally reads lidar poses from disk, ignore lidar_poses_chunk.
        # If it needs poses passed in, add it to the signature and pass it.
        wai_frames_chunk = run_parallel(
            camera_files=camera_files,
            lidar_files=lidar_files,
            camera_intrinsics=camera_intrinsics,
            camera_poses=camera_poses,
            target_scene_root=target_scene_root,
            indices=ids_chunk
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
            },
        }

        store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    CONVERSION_PATH = Path(__file__).parent
    cfg = argconf_parse(CONVERSION_PATH / "pandaset_config.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_pandaset_scene, cfg, get_original_scene_names_func=get_grand_tour_scene_names)
