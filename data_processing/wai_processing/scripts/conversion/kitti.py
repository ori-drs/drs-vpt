# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pykitti
from argconf import argconf_parse
from tqdm import tqdm
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
)

from wai_processing.utils.lidar import warp_cloud_es_ee_three
from drs_vpt._vendor.mapanything.utils.wai.core import store_data

wai_lock = threading.Lock()


def get_kitti_scene_names(cfg):
    root_path = os.path.join(cfg.original_root, "sequences")
    # Get all scene names to process
    scene_names = [x for x in sorted(os.listdir(root_path))
                   if os.path.isdir(os.path.join(root_path, x)) and int(x) < 9]
    return scene_names


def write_image_depth_map_and_cloud_metadata(dataset,
                                             target_scene_root, index, camname):
    cloud = dataset.get_velo(index)[:, :3]
    if camname == "2":
        undistorted_image = np.array(dataset.get_cam2(index))
        # Process point cloud
        T_cam_velo = dataset.calib.T_cam2_velo
        T_cam0_cam = dataset.calib.T_cam0_velo @ np.linalg.inv(dataset.calib.T_cam2_velo)
        K = dataset.calib.K_cam2
        T_base0_base = dataset.poses[index]
        cam2world = T_base0_base @ T_cam0_cam
    elif camname == "3":
        undistorted_image = np.array(dataset.get_cam3(index))
        # Process point cloud
        T_cam_velo = dataset.calib.T_cam3_velo
        T_cam0_cam = dataset.calib.T_cam0_velo @ np.linalg.inv(dataset.calib.T_cam3_velo)
        K = dataset.calib.K_cam3
        T_base0_base = dataset.poses[index]
        cam2world = T_base0_base @ T_cam0_cam

    undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB)
    new_shape = undistorted_image.shape[:2]
    full_cloud_in_camera_frame = warp_cloud_es_ee_three(T_cam_velo, cloud)
    mask = full_cloud_in_camera_frame[:, 2] > 0.0
    cloud_in_camera = full_cloud_in_camera_frame[mask]
    cloud_pixels = \
        np.round(
            cv2.projectPoints(cloud_in_camera, np.zeros(3), np.zeros(3), K, np.zeros(4))[
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
    output_prefix = f"{index:06d}_{camname}"

    rel_image_out_path = Path("images") / f"{output_prefix}.png"
    rel_depth_out_path = Path("depth") / f"{output_prefix}.exr"
    rel_cloud_out_path = Path("cloud") / f"{output_prefix}.npz"

    cv2.imwrite(os.path.join(target_scene_root, rel_image_out_path), undistorted_image)
    cv2.imwrite(os.path.join(target_scene_root, rel_depth_out_path), depth_image)
    np.savez(os.path.join(target_scene_root, rel_cloud_out_path), cloud_in_camera_frame=full_cloud_in_camera_frame)

    wai_frame = {
        "frame_name": output_prefix,
        "image": str(rel_image_out_path),
        "file_path": str(rel_image_out_path),
        "depth": str(rel_depth_out_path),
        "cloud": str(rel_cloud_out_path),
        "transform_matrix": cam2world.tolist(),
        "h": depth_image.shape[0],
        "w": depth_image.shape[1],
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }
    return wai_frame


def fill_all_wai_frames(kitti_root, target_scene_root, scene_name, indices):
    dataset = pykitti.odometry(kitti_root, scene_name)

    wai_frames = []
    with tqdm(total=len(indices) * 2, desc="Processing Stamps", dynamic_ncols=True, position=0) as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count() / 2) as executor:
            futures = []
            for index in indices:
                futures.append(executor.submit(
                    write_image_depth_map_and_cloud_metadata, dataset,
                    target_scene_root=target_scene_root, index=index,
                    camname="2"
                ))
            for index in indices:
                futures.append(executor.submit(
                    write_image_depth_map_and_cloud_metadata, dataset,
                    target_scene_root=target_scene_root, index=index,
                    camname="3"
                ))

            for future in as_completed(futures):
                wai_frame = future.result()
                with wai_lock:  # append safely
                    if wai_frame is not None:
                        wai_frames.append(wai_frame)
                pbar.update(1)
    return wai_frames


def process_kitti_scene(cfg, scene_name):
    kitti_root = Path(cfg.original_root)
    dataset = pykitti.odometry(kitti_root, scene_name)
    # The frames on disk (len(dataset) counts times.txt, which lists every frame of the sequence)
    n = min(len(dataset.cam2_files), len(dataset.cam3_files), len(dataset.velo_files))

    chunk_size = 100
    for chunk_idx, start in enumerate(range(0, n, chunk_size)):
        end = min(n-1, start+chunk_size)
        subscene_name = f"{scene_name}_{chunk_idx:04d}"
        target_scene_root = Path(f"{cfg.root}/{subscene_name}")

        image_dir = target_scene_root / "images"
        image_dir.mkdir(parents=True, exist_ok=False)
        depth_dir = target_scene_root / "depth"
        depth_dir.mkdir(parents=True, exist_ok=False)
        cloud_dir = target_scene_root / "cloud"
        cloud_dir.mkdir(parents=True, exist_ok=False)

        indices = list(range(start, end))
        wai_frames = fill_all_wai_frames(kitti_root, target_scene_root, scene_name, indices)

        # Construct overall scene metadata
        scene_meta = {
            "scene_name": scene_name,
            "dataset_name": cfg.dataset_name,
            "version": cfg.version,
            "shared_intrinsics": False,
            "camera_model": "PINHOLE",
            "camera_convention": "opencv",
            "scale_type": "metric",
            "scene_modalities": {},
            "frames": wai_frames,
            "frame_modalities": {
                "image": {"frame_key": "image", "format": "image"},
                "depth": {
                    "frame_key": "depth",
                    "format": "depth",
                },
                "cloud": {
                    "frame_key": "cloud",
                    "format": "cloud",
                },
            },
        }
        store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")




if __name__ == "__main__":
    CONVERSION_PATH = Path(__file__).parent
    cfg = argconf_parse(CONVERSION_PATH / "kitti_config.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_kitti_scene, cfg, get_original_scene_names_func=get_kitti_scene_names)
