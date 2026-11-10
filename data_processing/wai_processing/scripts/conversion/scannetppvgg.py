# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
- Loads vilens and image data from OSD or GrandTour
- Assumes vilens has run successfully and that the extrinsic and intrinsic calibrations are available
- Assumes there are exact matching timestamps between images and undistorted clouds

"""
import os
import shutil
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

def get_scannetppvgg_scene_names(
    cfg,
):
    scene_names = os.listdir(cfg.original_root)
    return scene_names


def process_stamp(image_prefix, K, T_world_cam, source_scene_root, target_scene_root):

    # Undistort and resize image
    image_path = os.path.join(source_scene_root, "images", f"{image_prefix}.jpg")
    depth_path = os.path.join(source_scene_root, "depth", f"{image_prefix}.png")
    depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depthmap = depthmap.astype(np.float32) / 1000.0

    # Save outputs
    output_prefix = image_prefix
    rel_target_image_path = Path("images") / f"{output_prefix}.jpg"
    shutil.copy(image_path, os.path.join(target_scene_root, rel_target_image_path))
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


def run_parallel(scene_root, scene_metadata, target_scene_root):
    wai_frames = []
    n_frames = len(scene_metadata["images"])
    with tqdm(total=n_frames, desc="Processing Stamps") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            images, intrinsincs, trajectories = (
                scene_metadata["images"], scene_metadata["intrinsics"], scene_metadata["trajectories"]
            )
            images = images[::10]
            intrinsincs = intrinsincs[::10]
            trajectories = trajectories[::10]
            for image_prefix, K, T_world_cam in zip(images, intrinsincs, trajectories):
                futures.append(executor.submit(
                    process_stamp, image_prefix, K, T_world_cam, scene_root, target_scene_root))

            for future in as_completed(futures):
                wai_frame = future.result()
                with wai_lock:  # append safely
                    wai_frames.append(wai_frame)
                pbar.update(1)

    return wai_frames


def process_scannetppvgg_scene(cfg, scene_name):
    """
    """
    scene_root = Path(f"{cfg.original_root}/{scene_name}")
    assert os.path.exists(scene_root)

    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    depth_dir = target_scene_root / "depth"
    depth_dir.mkdir(parents=True, exist_ok=False)

    original_scene_metadata = np.load(os.path.join(scene_root, "scene_metadata.npz"))

    wai_frames = run_parallel(scene_root, original_scene_metadata, target_scene_root)

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
    cfg = argconf_parse(CONVERSION_PATH / "scannetppvgg_config.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_scannetppvgg_scene, cfg, get_original_scene_names_func=get_scannetppvgg_scene_names)
