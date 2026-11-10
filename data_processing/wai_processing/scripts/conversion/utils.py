# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import glob
import json
import os

import numpy as np
import pyntcloud
from scipy.spatial.transform import Rotation


def list_argoverse_folders(base_path):
    """
    List files and subdirectories with full paths in dynamically changing ArgoVerse tracking train directories.

    Args:
        base_path (str): The base path where train directories are located.

    Returns:
        dict: A dictionary with train folder names as keys and lists of full paths as values.
    """
    train_dirs = sorted(glob.glob(os.path.join(base_path, "argoverse-tracking/train*")))  # Finds all trainX folders
    train_contents = {
        os.path.basename(d): [os.path.join(d, f) for f in os.listdir(d)]
        for d in train_dirs if os.path.isdir(d)
    }
    return train_contents


def load_car_pose_from_path(path):
    with open(path, "r") as f:
        data = json.load(f)
        T = np.eye(4)
        T[:3, -1] = data["translation"]
        q_wxyz = data["rotation"]
        q_xyzw = q_wxyz[1:] + q_wxyz[:1]
        T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
        return T


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_ply(ply_fpath) -> np.ndarray:
    """Load a point cloud file from a filepath.

    Args:
        ply_fpath: Path to a PLY file

    Returns:
        arr: Array of shape (N, 3)
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]

    return np.concatenate((x, y, z), axis=1)
