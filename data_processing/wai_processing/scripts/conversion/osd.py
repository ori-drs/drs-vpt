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

import numpy as np
from argconf import argconf_parse
from tqdm import tqdm

from wai_processing.utils.lidar import load_camera_files, load_undistorted_cloud_files, \
    load_base_frame_poses_csv, \
    load_osd_sensor_yaml, find_common_camera_lidar_stamps, warp_cloud_es_ee_three, read_cloud, \
    Distorter

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
)

wai_lock = threading.Lock()

from drs_vpt._vendor.mapanything.utils.wai.core import store_data


def get_osd_scene_names(
    cfg,
):
    # Get all scene names to process
    scene_names = [x for x in sorted(os.listdir(cfg.original_root))
                   if os.path.isdir(os.path.join(cfg.original_root, x))]
    return scene_names


def process_stamp(cam_name, camera_lidar_stamp_match, cameras_data,
                  T_base_lidar, vilens_poses, camera_files, lidar_files,
                  target_scene_root, image_dir):
    distorter = Distorter.from_config(cameras_data[cam_name])[1]
    T_base_camera = T_base_lidar @ cameras_data[cam_name]["T_lidar_cam"]
    T_camera_base = np.linalg.inv(T_base_camera)
    T_world_camera = vilens_poses[camera_lidar_stamp_match.lidar_stamp] @ T_base_camera
    image_path, cloud_path = (camera_files[cam_name][camera_lidar_stamp_match.camera_stamp],
                              lidar_files[camera_lidar_stamp_match.lidar_stamp])

    # Undistort and resize image
    distorted_image = cv2.imread(image_path)
    undistorted_image = distorter.undistort_image(distorted_image)
    rescale_factor = 1.0
    undistorted_image = cv2.resize(
        undistorted_image, None, fx=rescale_factor, fy=rescale_factor,
        interpolation=cv2.INTER_LANCZOS4
    )
    new_shape = undistorted_image.shape[:2]

    # Process point cloud
    cloud_in_camera = warp_cloud_es_ee_three(T_camera_base, read_cloud(cloud_path))
    cloud_in_camera = cloud_in_camera[cloud_in_camera[:, 2] > 0.0]
    cloud_pixels = np.round(
        cv2.projectPoints(cloud_in_camera, np.zeros(3), np.zeros(3),
                          distorter.new_K * rescale_factor, np.zeros(4))[0].squeeze()
    ).astype(int)
    validity_mask = (
        (cloud_pixels[:, 0] >= 0) & (cloud_pixels[:, 0] < new_shape[1]) &
        (cloud_pixels[:, 1] >= 0) & (cloud_pixels[:, 1] < new_shape[0])
    )
    cloud_pixels = cloud_pixels[validity_mask]
    cloud_in_camera = cloud_in_camera[validity_mask]
    x, y = cloud_pixels.T

    # Create depth map
    depth_image = np.zeros(new_shape, dtype=np.float32)
    depth_image[y, x] = cloud_in_camera[:, 2]

    H = depth_image.shape[0]
    top_valid = (depth_image[: H//2] > 0).sum() > 2000
    bottom_valid = (depth_image[-H//2:] > 0).sum() > 3000
    if not (top_valid and bottom_valid):
        return None

    # Save outputs
    output_prefix = f"{camera_lidar_stamp_match.camera_stamp}_{cam_name}"
    rel_target_image_path = Path("images") / f"{output_prefix}.jpg"
    cv2.imwrite(os.path.join(target_scene_root, rel_target_image_path), undistorted_image)
    rel_depth_out_path = Path("depth") / f"{output_prefix}.exr"
    cv2.imwrite(os.path.join(target_scene_root, rel_depth_out_path), depth_image)

    new_intrinsics = distorter.new_K * rescale_factor
    new_intrinsics[2, 2] = 1
    np.savez(
        os.path.join(image_dir, f"{output_prefix}.npz"),
        intrinsics=new_intrinsics,
        cam2world=T_world_camera,
    )

    # Prepare WAI metadata
    wai_frame = {
        "frame_name": output_prefix,
        "image": str(rel_target_image_path),
        "file_path": str(rel_target_image_path),
        "depth": str(rel_depth_out_path),
        "transform_matrix": T_world_camera.tolist(),
        "h": depth_image.shape[0],
        "w": depth_image.shape[1],
        "fl_x": float(new_intrinsics[0, 0]),
        "fl_y": float(new_intrinsics[1, 1]),
        "cx": float(new_intrinsics[0, 2]),
        "cy": float(new_intrinsics[1, 2]),
    }

    return wai_frame


def run_parallel(matched_camera_lidar_stamps, cameras_data, T_base_lidar,
                 vilens_poses, camera_files, lidar_files, target_scene_root,
                 image_dir, total_stamps):
    wai_frames = []
    with tqdm(total=total_stamps, desc="Processing Stamps") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for cam_name, matches in matched_camera_lidar_stamps.items():
                for match in matches:
                    futures.append(executor.submit(
                        process_stamp, cam_name, match,
                        cameras_data, T_base_lidar, vilens_poses,
                        camera_files, lidar_files, target_scene_root, image_dir
                    ))

            for future in as_completed(futures):
                wai_frame = future.result()
                with wai_lock:  # append safely
                    if wai_frame is not None:
                        wai_frames.append(wai_frame)
                pbar.update(1)

    return wai_frames

CHUNK_SIZE = 100

def chunk_indices(num_frames, chunk_size):
    for i in range(0, num_frames, chunk_size):
        yield range(i, min(i + chunk_size, num_frames))


def chunk_matched_camera_lidar_stamps_min_ref(matched_camera_lidar_stamps, chunk_size):
    """
    matched_camera_lidar_stamps: Dict[camera_name, List[stamp]]

    Uses the camera with the MIN number of stamps as reference.
    All other cameras are truncated to that length.
    Chunking is then index-aligned and simple.
    """

    # Pick reference camera = shortest sequence
    ref_cam = min(
        matched_camera_lidar_stamps,
        key=lambda cam: len(matched_camera_lidar_stamps[cam])
    )
    ref_len = len(matched_camera_lidar_stamps[ref_cam])

    # Truncate all cameras to ref_len
    truncated = {
        cam: list(matched_camera_lidar_stamps[cam])[:ref_len]
        for cam in matched_camera_lidar_stamps
    }

    # Chunk by index
    chunks = []
    for start in range(0, ref_len, chunk_size):
        end = start + chunk_size
        chunk = {
            cam: stamps[start:end]
            for cam, stamps in truncated.items()
        }
        chunks.append(chunk)

    return chunks

def process_osd_scene(cfg, scene_name):
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

    images_folder = os.path.join(scene_root, "raw/images")
    camera_files = load_camera_files(images_folder)
    camera_stamps = {k: sorted(list(v.keys())) for k, v in camera_files.items()}

    undistorted_clouds_folder = os.path.join(scene_root, "processed/vilens-slam/undist-clouds")
    lidar_files = load_undistorted_cloud_files(undistorted_clouds_folder)
    lidar_stamps = list(lidar_files.keys())

    vilens_poses_path = os.path.join(scene_root, "processed/vilens-slam/slam-poses.csv")
    vilens_poses = load_base_frame_poses_csv(vilens_poses_path)

    sensor_yaml_path = os.path.join(cfg.original_root, "sensor.yaml")
    cameras_data, T_base_lidar = load_osd_sensor_yaml(sensor_yaml_path)

    matched_camera_lidar_stamps = find_common_camera_lidar_stamps(camera_stamps, lidar_stamps)
    for k,v in matched_camera_lidar_stamps.items():
        matched_camera_lidar_stamps[k] = sorted(v)

    stamp_chunks = chunk_matched_camera_lidar_stamps_min_ref(
        matched_camera_lidar_stamps,
        CHUNK_SIZE,
    )

    for chunk_idx, stamps_chunk in enumerate(stamp_chunks):
        subscene_name = f"{scene_name}_{chunk_idx:04d}"
        target_scene_root = Path(f"{cfg.root}/{subscene_name}")

        image_dir = target_scene_root / "images"
        depth_dir = target_scene_root / "depth"

        image_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        total_stamps_chunk = sum(len(v) for v in stamps_chunk.values())

        wai_frames = run_parallel(
            stamps_chunk,
            cameras_data,
            T_base_lidar,
            vilens_poses,
            camera_files,
            lidar_files,
            target_scene_root,
            image_dir,
            total_stamps_chunk,
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
            "frames": wai_frames,
            "frame_modalities": {
                "image": {"frame_key": "image", "format": "image"},
                "depth": {"frame_key": "depth", "format": "depth"},
            },
        }

        store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    CONVERSION_PATH = Path(__file__).parent
    cfg = argconf_parse(CONVERSION_PATH / "osd_config.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_osd_scene, cfg, get_original_scene_names_func=get_osd_scene_names)
