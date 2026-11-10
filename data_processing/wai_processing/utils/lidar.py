# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

"""
LiDAR and camera helpers for the WAI conversion scripts: loading clouds, camera files, sensor calibration and poses,
matching camera and LiDAR timestamps, transforming clouds, and camera distortion models.
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, Set, List

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True, order=True)
class CameraLiDARMatch:
    camera_stamp: int
    lidar_stamp: int


def parse_timestamp(filename):
    """
    Extract and standardize timestamps from filenames.

    Supported formats:
    - 1710252325.945682506.jpg
    - 1710252304_395841875.jpg

    Returns:
        Tuple (seconds, nanoseconds)
    """
    match = re.search(r"(\d{10})[._](\d+)", filename)
    if match:
        seconds = int(match.group(1))
        nanoseconds = int(match.group(2).ljust(9, '0'))  # Ensure nanoseconds are 9 digits
        return seconds * int(1e9) + nanoseconds
    return None  # Return None if no match found


def load_camera_files(images_root: str) -> Dict[str, Dict[int, str]]:
    """
    Loads and parses camera timestamps from image directories.

    This function organizes image file timestamps for multiple camera subdirectories
    (called 'cam0', 'cam1', 'cam2') located under the provided root directory.
    It creates a mapping of parsed timestamps to the file path for each camera directory.

    Parameters:
    images_root: A string representing the root directory containing camera subdirectories.

    Returns:
    A dictionary where keys are camera subdirectory names (e.g., 'cam0') and values
    are dictionaries mapping parsed timestamps to their corresponding file paths.
    """
    subdirs = [f"cam{x}" for x in range(3)]
    camera_stamps = dict()
    for x in subdirs:
        folder = os.path.join(images_root, x)
        stamp_to_file_path = {parse_timestamp(x): str(os.path.join(folder, x)) for x in os.listdir(folder)}
        camera_stamps[x] = stamp_to_file_path
    return camera_stamps


def load_undistorted_cloud_files(clouds_root: str) -> Dict[int, str]:
    """
    Loads undistorted point cloud timestamp file paths from a specified root directory.

    Parameters:
    clouds_root (str): The root directory containing point cloud files.

    Returns:
    Dict[int, str]: A dictionary where keys are parsed timestamps and values are the file paths of the corresponding point cloud files.
    """
    stamps = {parse_timestamp(x): os.path.join(clouds_root, x) for x in os.listdir(clouds_root)}
    return stamps


def load_base_frame_poses_csv(filepath):
    """
    Load vilens poses csv and convert it into a dictionary mapping timestamps to SE(3) transformation matrices.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        dict: { timestamp (int): 4x4 SE(3) matrix (np.ndarray) }
    """
    # Read and clean CSV
    with open(filepath, "r") as f:
        first_line = f.readline().lstrip("#").strip()  # Strip '#' and spaces from header

    df = pd.read_csv(filepath, comment="#", header=None)  # Skip commented lines
    df.columns = [col.strip() for col in first_line.split(",")]  # Clean column names

    df["sec"] = df["sec"].astype("uint64")  # Convert to unsigned 64-bit integer
    df["nsec"] = df["nsec"].astype("uint64")
    df["timestamp"] = (df["sec"] * 10 ** 9 + df["nsec"]).astype("uint64")

    # Construct SE(3) matrices
    se3_dict = {}
    for idx in df.index:
        timestamp = df.at[idx, "timestamp"]  # Explicitly index timestamp column

        # Translation vector
        t = np.array([df.at[idx, "x"], df.at[idx, "y"], df.at[idx, "z"]])

        # Quaternion to rotation matrix
        quat = [df.at[idx, "qx"], df.at[idx, "qy"], df.at[idx, "qz"], df.at[idx, "qw"]]
        R_mat = R.from_quat(quat).as_matrix()  # 3x3 rotation matrix

        # Construct 4x4 SE(3) matrix
        T = np.eye(4)
        T[:3, :3] = R_mat  # Set rotation
        T[:3, 3] = t  # Set translation

        # Store in dictionary
        se3_dict[int(timestamp)] = T  # Convert uint64 to Python int

    return se3_dict


def find_common_camera_lidar_stamps(camera_stamps: Dict[str, List[int]],
                                    lidar_stamps: List[int], slack_in_secs: [float] = 0.02) -> Dict[
    str, Set[CameraLiDARMatch]]:
    lidar_stamps = np.array(lidar_stamps)
    matches = {k: set() for k in camera_stamps.keys()}
    for k, v in camera_stamps.items():
        for camera_stamp in v:
            cam_lidar_delta = np.abs(camera_stamp * 1e-9 - lidar_stamps * 1e-9)
            if np.min(cam_lidar_delta) < slack_in_secs:
                lidar_stamp_index = np.argmin(cam_lidar_delta)
                matches[k].add(CameraLiDARMatch(camera_stamp, lidar_stamps[lidar_stamp_index].item()))
    return matches


def extract_camera_id(topic):
    """
    Extracts the camera ID (e.g., cam0, cam1, cam2) from a ROS topic string.

    Args:
        topic (str): The ROS topic string.

    Returns:
        str: The extracted camera ID.
    """
    parts = topic.split("/")
    for part in parts:
        if part.startswith("cam"):
            return part  # Return the first occurrence of "camX"

    return None  # Return None if not found


def es_ee_three_from_xyz_quat(T_base_lidar_t_xyz_q_xyzw):
    R_b__l = R.from_quat(T_base_lidar_t_xyz_q_xyzw[-4:]).as_matrix()
    t_b__l = np.array(T_base_lidar_t_xyz_q_xyzw[:3])
    T_b__l = np.eye(4)
    T_b__l[:3, :3] = R_b__l
    T_b__l[:3, -1] = t_b__l
    return T_b__l


def load_osd_sensor_yaml(path):
    with open(path, "r") as f:
        sensor_data = yaml.safe_load(f)["sensor"]
        cameras_data = sensor_data["cameras"]
        cam_names = [extract_camera_id(x["topic"]) for x in cameras_data]
        cameras_data = {k: v for k, v in zip(cam_names, cameras_data)}
        for k in cameras_data:
            cameras_data[k]["resolution"] = cameras_data[k]["image_width"], cameras_data[k]["image_height"]
            cameras_data[k]["rostopic"] = cameras_data[k]["topic"]
            cameras_data[k]["distortion_coeffs"] = cameras_data[k]["extra_params"]
            cameras_data[k]["distortion_model"] = "equidistant"
            cameras_data[k]["T_cam_lidar"] = es_ee_three_from_xyz_quat(
                cameras_data[k]["T_cam_lidar_t_xyz_q_xyzw"])
            cameras_data[k]["T_lidar_cam"] = np.linalg.inv(cameras_data[k]["T_cam_lidar"])
            del cameras_data[k]["topic"]
            del cameras_data[k]["extra_params"]
            del cameras_data[k]["image_width"]
            del cameras_data[k]["image_height"]
            del cameras_data[k]["T_cam_lidar_t_xyz_q_xyzw"]
            if "T_cam_imu_t_xyz_q_xyzw" in cameras_data[k]:
                del cameras_data[k]["T_cam_imu_t_xyz_q_xyzw"]

        T_base_lidar_t_xyz_q_xyzw = sensor_data["T_base_lidar_t_xyz_q_xyzw"]
        T_b__l = es_ee_three_from_xyz_quat(T_base_lidar_t_xyz_q_xyzw)

    return cameras_data, T_b__l


def warp_cloud_es_ee_three(T_se3, cloud_xyz):
    """
    Applies a 4x4 SE(3) transformation matrix to an Nx3 array of XYZ points.

    Args:
        T_se3 (np.ndarray): 4x4 SE(3) transformation matrix.
        cloud_xyz (np.ndarray): Nx3 array of 3D points.

    Returns:
        np.ndarray: Transformed Nx3 point cloud.
    """
    # Convert Nx3 to Nx4 homogeneous coordinates
    ones = np.ones((cloud_xyz.shape[0], 1))  # Column of ones
    cloud_homogeneous = np.hstack((cloud_xyz, ones))  # Nx4 matrix

    # Apply the SE(3) transformation (T * X)
    transformed_homogeneous = (T_se3 @ cloud_homogeneous.T).T  # Matrix multiplication

    # Convert back to Nx3 by removing the homogeneous coordinate
    transformed_xyz = transformed_homogeneous[:, :3]

    return transformed_xyz


def read_cloud(cloud_path):
    """
    Reads a PCD file and extracts the XYZ points as a NumPy array.

    Args:
        cloud_path (str): Path to the PCD file.

    Returns:
        np.ndarray: Nx3 array of point coordinates (X, Y, Z).
    """
    pcd = o3d.io.read_point_cloud(cloud_path)  # Load point cloud
    points = np.asarray(pcd.points)  # Convert to NumPy array (Nx3)
    return points


def K_from_fxfycxcy(intrinsics):
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


class Distorter(object):
    @classmethod
    def from_config_path(cls, config_path: str):
        with open(config_path, "r") as f:
            all_camera_configs = yaml.safe_load(f)
            return [Distorter.from_config(all_camera_configs[x]) for x in all_camera_configs]

    @classmethod
    def from_config(cls, config):
        assert "rostopic" in config
        distortion_model = config['distortion_model']
        intrinsics = config['intrinsics']
        D = np.array(config['distortion_coeffs'])
        resolution = config['resolution']
        K = K_from_fxfycxcy(intrinsics)
        if distortion_model == "radtan":
            return config["rostopic"], RadTanDistorter(K=K, dist_params=D, image_size=resolution)
        elif distortion_model == "equidistant":
            return config["rostopic"], FisheyeDistorter(K=K, dist_params=D, image_size=resolution)
        else:
            return NotImplementedError

    def __init__(self, K, D, image_size):
        super().__init__()
        self.mtx = K
        self.dist = D
        self.image_size = image_size
        self.new_K = None

    def distort(self, batched_points2d):
        raise NotImplementedError

    def undistort(self, batched_points2d):
        raise NotImplementedError

    def undistort_image(self, distorted_image):
        raise NotImplementedError

    def project(self, batched_points3d):
        raise NotImplementedError

    def project_undistorted(self, batched_points3d):
        raise NotImplementedError


class FisheyeDistorter(Distorter):
    def __init__(self, K, dist_params, image_size):
        super().__init__(K, dist_params, image_size)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.mtx.copy(), self.dist.copy(),
                                                                       self.image_size,
                                                                       np.eye(3), balance=0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.mtx, self.dist, np.eye(3), new_K, self.image_size,
                                                         cv2.CV_16SC2)
        self.new_K = new_K
        self.map1 = map1
        self.map2 = map2

    def undistort(self, batched_points2d):
        return cv2.fisheye.undistortPoints(batched_points2d, self.mtx, self.dist)

    def distort(self, batched_points2d):
        return cv2.fisheye.distortPoints(batched_points2d, self.mtx, self.dist)

    def project(self, batched_points3d):
        return cv2.fisheye.projectPoints(batched_points3d, np.zeros(3), np.zeros(3),
                                         self.mtx, self.dist)[0]

    def project_undistorted(self, batched_points3d):
        return cv2.projectPoints(batched_points3d, np.zeros(3), np.zeros(3),
                                 self.new_K, np.zeros(4))[0]

    def undistort_image(self, distorted_image):
        return cv2.remap(distorted_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


class RadTanDistorter(Distorter):
    def __init__(self, K, dist_params, image_size):
        super().__init__(K, dist_params, image_size)
        """
        xdistorted = x * (1 + k1 * r2 + k2 * r4 + k3 * r6)
        """
        self.new_K, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, image_size, alpha=0, newImgSize=image_size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.mtx, self.dist, np.eye(3), self.new_K, self.image_size,
                                                           cv2.CV_16SC2)

    def distort(self, batched_points2d):
        return cv2.distortPoints(batched_points2d, self.mtx, self.dist)

    def undistort(self, batched_points2d):
        return cv2.undistortPoints(batched_points2d, self.mtx, self.dist)

    def undistort_image(self, distorted_image):
        return cv2.remap(distorted_image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def project(self, batched_points3d):
        return cv2.projectPoints(batched_points3d, np.zeros(3), np.zeros(3),
                                 self.mtx, self.dist)[0]

    def project_undistorted(self, batched_points3d):
        return cv2.projectPoints(batched_points3d, np.zeros(3), np.zeros(3),
                                 self.new_K, np.zeros(4))[0]


class NoDistorter(Distorter):
    def __init__(self, K, D, image_size):
        super().__init__(K, D, image_size)

    def distort(self, batched_points2d):
        return batched_points2d

    def undistort(self, batched_points2d):
        return batched_points2d
