# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
- Loads vilens and image data from OSD or GrandTour
- Assumes vilens has run successfully and that the extrinsic and intrinsic calibrations are available
- Assumes there are exact matching timestamps between images and undistorted clouds

"""
import glob
import os
import random
import threading
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from argconf import argconf_parse
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from wai_processing.scripts.conversion.utils import load_ply, load_car_pose_from_path, load_json
from wai_processing.utils.lidar import find_common_camera_lidar_stamps, warp_cloud_es_ee_three

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from wai_processing.utils.wrapper import (
    convert_scenes_wrapper,
)

wai_lock = threading.Lock()

from drs_vpt._vendor.mapanything.utils.wai.core import store_data


def get_argoverse_scene_names(
    cfg,
):
    train_dirs = glob.glob(f"{cfg.original_root}/**/train*/", recursive=True)
    scene_names = []
    for dir in train_dirs:
        scene_names += os.listdir(dir)
    return scene_names


def chunk_matched_camera_lidar_stamps_max_ref(matched_camera_lidar_stamps, chunk_size):
    """
    matched_camera_lidar_stamps: Dict[camera_name, List[stamp]]

    Uses the camera with the MAX number of stamps as reference.
    Other cameras may have shorter sequences.
    Chunking is index-aligned; shorter cameras yield shorter chunks.
    """

    # Pick reference camera = longest sequence
    ref_cam = max(
        matched_camera_lidar_stamps,
        key=lambda cam: len(matched_camera_lidar_stamps[cam])
    )
    ref_len = len(matched_camera_lidar_stamps[ref_cam])

    # Chunk by index up to ref_len
    chunks = []
    for start in range(0, ref_len, chunk_size):
        end = start + chunk_size
        chunk = {
            cam: stamps[start:end]
            for cam, stamps in matched_camera_lidar_stamps.items()
        }
        chunks.append(chunk)

    return chunks


def process_argoverse_stamp(args):
    (cam_name, camera_stamp_lidar_stamp, camera_calibration,
     matched_poses, camera_files, lidar_files,
     sequence_folder, output_root) = args
    T_city_ego_camt, T_city_ego_lidart = matched_poses[cam_name][camera_stamp_lidar_stamp]
    lidar_file = lidar_files[camera_stamp_lidar_stamp.lidar_stamp]
    image_path = camera_files[cam_name][camera_stamp_lidar_stamp.camera_stamp]
    T_city_cam = T_city_ego_camt @ np.linalg.inv(camera_calibration[cam_name]["T_cam_car"])

    T_cam_lidar = np.linalg.inv(T_city_cam) @ T_city_ego_lidart
    K = camera_calibration[cam_name]["K"]
    undistorted_image = cv2.imread(image_path)
    rescale_factor = 1080 / max(undistorted_image.shape[:2])
    undistorted_image = cv2.resize(undistorted_image, None, fx=rescale_factor, fy=rescale_factor,
                                   interpolation=cv2.INTER_LANCZOS4)
    new_shape = undistorted_image.shape[:2]

    # Process point cloud
    cloud_in_camera_frame = warp_cloud_es_ee_three(T_cam_lidar, load_ply(lidar_file))
    cloud_in_camera = cloud_in_camera_frame[cloud_in_camera_frame[:, 2] > 0.0]
    cloud_pixels = \
        np.round(
            cv2.projectPoints(cloud_in_camera, np.zeros(3), np.zeros(3), rescale_factor * K, np.zeros(4))[
                0].squeeze()).astype(int)
    validity_mask = (cloud_pixels[:, 0] >= 0) & (cloud_pixels[:, 0] < new_shape[1]) & (cloud_pixels[:, 1] >= 0) & (
        cloud_pixels[:, 1] < new_shape[0])
    cloud_pixels = cloud_pixels[validity_mask]
    cloud_in_camera = cloud_in_camera[validity_mask]
    x, y = cloud_pixels.T

    # Create depth map
    depth_image = np.zeros(new_shape, dtype=np.float32)
    depth_image[y, x] = cloud_in_camera[:, 2]
    valid_depth = depth_image > 0
    if valid_depth.sum() < 5000:
        return None

    # Save outputs
    target_scene_root = os.path.join(output_root)
    os.makedirs(output_root, exist_ok=True)
    output_prefix = f"{cam_name}_{camera_stamp_lidar_stamp.camera_stamp}"

    rel_target_image_path = Path("images") / f"{output_prefix}.jpg"
    os.makedirs(os.path.join(target_scene_root, "images"), exist_ok=True)
    cv2.imwrite(os.path.join(target_scene_root, rel_target_image_path), undistorted_image)

    rel_depth_out_path = Path("depth") / f"{output_prefix}.exr"
    os.makedirs(os.path.join(target_scene_root, "depth"), exist_ok=True)
    cv2.imwrite(os.path.join(target_scene_root, rel_depth_out_path), depth_image)

    rel_cloud_out_path = Path("cloud") / f"{output_prefix}.npz"
    os.makedirs(os.path.join(target_scene_root, "cloud"), exist_ok=True)
    store_data(os.path.join(target_scene_root, rel_cloud_out_path),
               cloud_in_camera_frame, "cloud")
    # Prepare WAI metadata
    wai_frame = {
        "frame_name": output_prefix,
        "image": str(rel_target_image_path),
        "file_path": str(rel_target_image_path),
        "depth": str(rel_depth_out_path),
        "cloud": str(rel_cloud_out_path),
        "transform_matrix": T_city_cam.tolist(),
        "h": depth_image.shape[0],
        "w": depth_image.shape[1],
        "fl_x": float(K[0, 0] * rescale_factor),
        "fl_y": float(K[1, 1] * rescale_factor),
        "cx": float(K[0, 2] * rescale_factor),
        "cy": float(K[1, 2] * rescale_factor),
    }

    return wai_frame


def process_matches(
    matched_stamps,
    car_poses_at_matched_times,
    camera_files,
    lidar_files,
    camera_calibration,
    sequence_name,
    output_root,
):
    total_stamps = sum(len(timestamps) for timestamps in matched_stamps.values())

    wai_frames = []  # <-- collect results

    with tqdm(total=total_stamps, desc="Processing Stamps") as pbar:
        with Pool(processes=8) as pool:
            for wai_frame in pool.imap_unordered(
                process_argoverse_stamp,
                [
                    (
                            cam_name,
                            camera_lidar_stamp_match,
                            camera_calibration,
                            car_poses_at_matched_times,
                            camera_files,
                            lidar_files,
                            sequence_name,
                            output_root,
                    )
                    for cam_name, camera_lidar_stamp_matches in matched_stamps.items()
                    for camera_lidar_stamp_match in camera_lidar_stamp_matches
                ],
            ):
                if wai_frame is not None:
                    wai_frames.append(wai_frame)  # <-- capture return
                pbar.update(1)

    return wai_frames


def process_argoverse_scene(cfg, scene_name):
    """
    Process a BlendedMVS scene into the WAI format.
    The PFM format depth maps are converted to default WAI depth format (exr).

    Expected input structure (Argoverse):
        scene-level
        ├── lidar
        ├── per_sweep_annotations_amodal
        ├── poses
        ├── ring_front_center
        ├── ring_front_left
        ├── ring_front_right
        ├── ring_rear_left
        ├── ring_rear_right
        ├── ring_side_left
        ├── ring_side_right
        ├── stereo_front_left
        ├── stereo_front_right
        └── track_labels_amodal

    """
    scene_matches = glob.glob(f"{cfg.original_root}/**/{scene_name}", recursive=True)
    assert len(scene_matches) == 1
    scene_root = Path(scene_matches[0])
    sequence_path = scene_root
    calibration_data = load_json(os.path.join(sequence_path, "vehicle_calibration_info.json"))
    camera_names = [x["key"].replace("image_raw_", "") for x in calibration_data["camera_data_"]]
    camera_files = {x: {int((y).replace(".jpg", "").replace(f"{x}_", "")):
                            os.path.join(sequence_path, x, y) for y in os.listdir(os.path.join(sequence_path, x))}
                    for x
                    in camera_names}
    camera_stamps = {k: list(v.keys()) for k, v in camera_files.items()}

    lidar_files = {int(x.replace("PC_", "").replace(".ply", "")): os.path.join(sequence_path, "lidar", x) for x in
                   os.listdir(os.path.join(sequence_path, "lidar"))}
    lidar_stamps = list(lidar_files.keys())

    for k, v in camera_stamps.items():
        for stamp in v:
            pose_path = os.path.join(sequence_path, f"poses/city_SE3_egovehicle_{stamp}.json")
            if not os.path.exists(pose_path):
                del camera_files[k][stamp]
                camera_stamps[k].remove(stamp)
    for stamp in lidar_stamps:
        pose_path = os.path.join(sequence_path, f"poses/city_SE3_egovehicle_{stamp}.json")
        if not os.path.exists(pose_path):
            lidar_stamps.remove(stamp)

    matched_camera_lidar_stamps = find_common_camera_lidar_stamps(camera_stamps, lidar_stamps, slack_in_secs=0.05)
    for k, v in matched_camera_lidar_stamps.items():
        matched_camera_lidar_stamps[k] = sorted(v)

    car_poses_at_matched_times = dict()
    for cam_name, matches in matched_camera_lidar_stamps.items():
        car_poses_at_matched_times[cam_name] = dict()
        for match in matches:
            car_poses_at_matched_times[cam_name][match] = (load_car_pose_from_path(
                os.path.join(sequence_path, "poses", f"city_SE3_egovehicle_{match.camera_stamp}.json")),
                                                           load_car_pose_from_path(
                                                               os.path.join(sequence_path, "poses",
                                                                            f"city_SE3_egovehicle_{match.lidar_stamp}.json")))
    camera_calibration_data = dict()
    for v in calibration_data["camera_data_"]:
        cam_name = v["key"].replace("image_raw_", "")
        fx = v["value"]["focal_length_x_px_"]
        fy = v["value"]["focal_length_y_px_"]
        cx = v["value"]["focal_center_x_px_"]
        cy = v["value"]["focal_center_y_px_"]
        T_cam_car = np.eye(4)
        T_cam_car[:3, -1] = v["value"]["vehicle_SE3_camera_"]["translation"]
        q_wxyz = v["value"]["vehicle_SE3_camera_"]["rotation"]["coefficients"]
        q_xyzw = q_wxyz[1:] + q_wxyz[:1]
        T_cam_car[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
        camera_calibration_data[cam_name] = dict()
        camera_calibration_data[cam_name]["K"] = np.array([[fx, 0.0, cx],
                                                           [0.0, fy, cy],
                                                           [0.0, 0.0, 1.0]
                                                           ])
        camera_calibration_data[cam_name]["T_cam_car"] = np.linalg.inv(T_cam_car)

    for k in matched_camera_lidar_stamps:
        trajectory = np.array([x[0][:3, -1] for x in car_poses_at_matched_times[k].values()])
        travels = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
        is_moving = travels > 0.1
        is_moving = np.insert(is_moving, 0, 1)
        matched_camera_lidar_stamps[k] = [x for i, x in enumerate(matched_camera_lidar_stamps[k]) if is_moving[i]]
        N = 90
        if len(matched_camera_lidar_stamps[k]) > N:
            # Seeded per scene and camera so a rerun keeps the same frames.
            rng = random.Random(f"{cfg.get('seed', 0)}-{scene_name}-{k}")
            matched_camera_lidar_stamps[k] = sorted(rng.sample(list(matched_camera_lidar_stamps[k]), N))

    CHUNK_SIZE = 30
    stamp_chunks = chunk_matched_camera_lidar_stamps_max_ref(
        matched_camera_lidar_stamps,
        CHUNK_SIZE,
    )

    for chunk_idx, stamps_chunk in enumerate(stamp_chunks):
        subscene_name = f"{scene_name}_{chunk_idx:04d}"
        target_scene_root = Path(f"{cfg.root}/{subscene_name}")

        image_dir = target_scene_root / "images"
        depth_dir = target_scene_root / "depth"
        cloud_dir = target_scene_root / "cloud"

        image_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)
        cloud_dir.mkdir(parents=True, exist_ok=True)

        wai_frames = process_matches(stamps_chunk,
                                     car_poses_at_matched_times,
                                     camera_files, lidar_files, camera_calibration_data,
                                     os.path.basename(sequence_path),
                                     target_scene_root)

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
                "cloud": {"frame_key": "cloud", "format": "cloud"},
            },
        }

        store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    CONVERSION_PATH = Path(__file__).parent
    cfg = argconf_parse(CONVERSION_PATH / "argoverse_config.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_argoverse_scene, cfg,
                           get_original_scene_names_func=get_argoverse_scene_names)
