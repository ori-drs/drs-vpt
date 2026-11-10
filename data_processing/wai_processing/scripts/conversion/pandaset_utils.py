# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import json
import os
import pickle
from typing import Dict

import numpy as np


def load_pandaset_camera_files(images_root: str) -> Dict[str, Dict[int, str]]:
    """
    Loads and parses camera ids from image directories.

    This function organizes image file timestamps for multiple camera subdirectories
    (called 'cam0', 'cam1', 'cam2') located under the provided root directory.
    It creates a mapping of parsed timestamps to the file path for each camera directory.

    Parameters:
    images_root: A string representing the root directory containing camera subdirectories.

    Returns:
    A dictionary where keys are camera subdirectory names (e.g., 'cam0') and values
    are dictionaries mapping parsed timestamps to their corresponding file paths.
    """
    subdirs = os.listdir(images_root)
    camera_ids = dict()
    for subdir_name in subdirs:
        folder = os.path.join(images_root, subdir_name)
        stamp_to_file_path = {int(x.split(".")[0]): str(os.path.join(folder, x)) for x in os.listdir(folder)
                              if ".jpg" in x}
        camera_ids[subdir_name] = stamp_to_file_path
    return camera_ids


def read_pandaset_cloud(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        cloud = data[["x", "y", "z"]].to_numpy()
        return cloud

def read_pandaset_cloud_with_intensity(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        cloud = data[["x", "y", "z", "i"]].to_numpy()
        return cloud

def load_pandaset_json_poses(path):
    from scipy.spatial.transform import Rotation as R
    poses = list()
    with open(path, "r") as f:
        data = json.load(f)
        for pose_data in data:
            t = np.array([pose_data["position"][field] for field in ["x", "y", "z"]])
            quat = np.array([pose_data["heading"][field] for field in ["x", "y", "z", "w"]])
            pose = np.eye(4)
            pose[:3, :3] = R.from_quat(quat).as_matrix()
            pose[:3, -1] = t
            poses.append(pose)
    return poses


def load_pandaset_json_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
        K = np.array([[data["fx"], 0.0, data["cx"]],
                      [0.0, data["fy"], data["cy"]],
                      [0.0, 0.0, 1.0]
                      ])
    return K
