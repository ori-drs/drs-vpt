# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
- Loads vilens and image data from OSD or GrandTour
- Assumes vilens has run successfully and that the extrinsic and intrinsic calibrations are available
- Assumes there are exact matching timestamps between images and undistorted clouds

"""
import glob
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from tqdm import tqdm


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
)

wai_lock = threading.Lock()

from drs_vpt._vendor.mapanything.utils.wai.core import store_data

def get_wildrgbd_scene_names(
    cfg,
):
    train_dirs = glob.glob(f"{cfg.original_root}/**/scene_*/", recursive=True)
    scene_names = []
    for dir in train_dirs:
        object_name = dir.split("/scenes/")[0].split("/")[-1]
        subscene_name = dir.split("/scenes/")[-1].split("/")[0]
        scene_names.append(f"{object_name}_{subscene_name}")
    return scene_names

def get_osd_scene_names(
    cfg,
):
    # Get all scene names to process
    scene_names = [x for x in sorted(os.listdir(cfg.original_root))
                   if os.path.isdir(os.path.join(cfg.original_root, x))]
    return scene_names


def process_stamp(file_prefix, source_scene_root, target_scene_root):

    # Undistort and resize image
    image_path = os.path.join(source_scene_root, "rgb", f"{file_prefix}.jpg")
    undistorted_image = cv2.imread(image_path)
    depth_path = os.path.join(source_scene_root, "depth", f"{file_prefix}.png")
    depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depthmap = depthmap.astype(np.float32) / 1000.0

    metadata_path = os.path.join(source_scene_root, "metadata", f"{file_prefix}.npz")
    metadata = np.load(metadata_path)
    K = metadata["camera_intrinsics"]
    T_world_cam = metadata["camera_pose"]

    # Save outputs
    output_prefix = file_prefix
    rel_target_image_path = Path("images") / f"{output_prefix}.png"
    cv2.imwrite(os.path.join(target_scene_root, rel_target_image_path), undistorted_image)
    rel_depth_out_path = Path("depth") / f"{output_prefix}.exr"
    cv2.imwrite(os.path.join(target_scene_root, rel_depth_out_path), depthmap)

    # Prepare WAI metadata
    wai_frame = {
        "frame_name": output_prefix,
        "image": str(rel_target_image_path),
        "file_path": str(rel_target_image_path),
        "depth": str(rel_depth_out_path),
        "transform_matrix": T_world_cam.tolist(),
        "h": depthmap.shape[0],
        "w": depthmap.shape[1],
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }

    return wai_frame


def run_parallel(scene_root, file_prefixes, target_scene_root):
    wai_frames = []
    with tqdm(total=len(file_prefixes), desc="Processing Stamps") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for file_prefix in file_prefixes:
                futures.append(executor.submit(
                    process_stamp, file_prefix, scene_root, target_scene_root))

            for future in as_completed(futures):
                wai_frame = future.result()
                with wai_lock:  # append safely
                    wai_frames.append(wai_frame)
                pbar.update(1)

    return wai_frames


def process_wildrgbd_scene(cfg, scene_name):
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
    object_name = scene_name.split("_")[0]
    subscene_name = scene_name.split(f"{object_name}_")[-1]
    scene_root = Path(f"{cfg.original_root}/{object_name}/scenes/{subscene_name}")
    assert os.path.exists(scene_root)

    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)

    images_folder = os.path.join(scene_root, "rgb")
    file_prefixes = [x.split(".")[0] for x in os.listdir(images_folder)]

    wai_frames = run_parallel(scene_root, file_prefixes, target_scene_root)

    # Construct overall scene metadata
    scene_meta = {
        "scene_name": scene_name,
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
            "depth": {
                "frame_key": "depth",
                "format": "depth",
            },
        },
    }
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    CONVERSION_PATH = Path(__file__).parent
    cfg = argconf_parse(CONVERSION_PATH / "wildrgbd_config.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_wildrgbd_scene, cfg, get_original_scene_names_func=get_wildrgbd_scene_names)
