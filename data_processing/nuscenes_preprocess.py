# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Turn a nuScenes download into the per-sample layout that wai_processing/scripts/conversion/nuscenes.py reads.

    pip install nuscenes-devkit
    python data_processing/nuscenes_preprocess.py --dataroot /raw/nuscenes --version v1.0-trainval \\
        --out /data/nuscenes_processed/train

This follows the image-to-point-cloud protocol of CorrI2P / VP2P on nuScenes:
  1. Index: LiDAR frames from the daytime scenes (every --frame_skip-th sweep, see lidar_frames), each with the front-camera
     frames whose ego pose is within --max_translation metres of the LiDAR's.
  2. Samples: for each indexed LiDAR frame, the sweep plus --accum_frames sweeps on either side (taking every
     --accum_skip-th), with points on the ego car removed, moved into the frame of one of its nearby camera frames
     (picked at random). Frames with fewer than --min_points points, or at most --min_in_image of them in the image,
     are skipped.

Each kept sample i is written as <out>/{img,K,PC,T_world_cam,P}/<i:06d>.{png,npy}:
    img           the camera image
    K             3x3 intrinsics (float32)
    PC            4 x N: x, y, z in the camera frame and intensity (float32)
    T_world_cam   4x4 camera-to-world pose (float32)
    P             4x4 LiDAR-to-camera transform (float32)
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm

# Night scenes of the v1.0-test split whose description does not say so.
UNLABELLED_NIGHT_SCENES = {
    "e59a4d0cc6a84ed59f78fb21a45cdcb4", "7209495d06f24712a063ac6c4a9b403b", "3d776ea805f240bb925bd9b50b258416",
    "48f81c548d0148fc8010a73d70b2ef9c", "2ab683f384234dce89800049dec19a30", "7edca4c44eac4f52a3105e1794e56b7e",
    "81c939ce8c0d4cc7b159cb5ed4c4e712", "24e6e64ecf794be4a51f7454c8b6d0b2", "828ed34a5e0c456fbf0751cabbab3341",
    "edfd6cfd1805477fbeadbd29f39ed599", "7692a3e112b44b408d191e45954a813c", "58d27a9f83294d99a4ff451dcad5f4d2",
    "a1573aef0bf74324b373dd8a22b4dd68", "ba06095d4e2e425b8e398668abc301d8", "7c315a1db2ac49439d281605f3cca6be",
    "732d7a84353f4ada803a9a115728496c", "1630a1d9cf8a46b3843662a23126e3f6", "f437809584344859882bdff7f8784c43",
}

EGO_CAR_HALF_EXTENT = (0.8, 2.7)  # |x| and |y| below these, in the LiDAR frame, are points on the car


def to_matrix(rotation, translation) -> np.ndarray:
    """4x4 float32 transform from a nuScenes quaternion and translation."""
    T = np.identity(4, dtype=np.float32)
    T[0:3, 0:3] = np.asarray(Quaternion(rotation).rotation_matrix).astype(np.float32)
    T[0:3, 3] = np.asarray(translation).astype(np.float32)
    return T


def ego_pose(nusc, sample_data) -> np.ndarray:
    """Ego-to-world transform at the sample_data's timestamp."""
    p = nusc.get("ego_pose", sample_data["ego_pose_token"])
    return to_matrix(p["rotation"], p["translation"])


def sensor_to_ego(nusc, sample_data) -> np.ndarray:
    c = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    return to_matrix(c["rotation"], c["translation"])


# ---------------------------------------------------------------------------
# 1. Index
# ---------------------------------------------------------------------------

def daytime_scenes(nusc) -> list:
    return [s["token"] for s in nusc.scene
            if "night" not in s["description"] and "Night" not in s["description"]
            and s["token"] not in UNLABELLED_NIGHT_SCENES]


def lidar_frames(nusc, scene_token: str, frame_skip: int) -> list:
    """LIDAR_TOP sample_data tokens of a scene: the first sweep, then sweeps 9, 19, 29, ... for frame_skip=10
    (the n-th sweep, counting the first as 1, is kept when n % frame_skip == 0)."""
    scene = nusc.get("scene", scene_token)
    lidar = nusc.get("sample_data", nusc.get("sample", scene["first_sample_token"])["data"]["LIDAR_TOP"])
    tokens, n = [lidar["token"]], 1
    while lidar["next"]:
        lidar = nusc.get("sample_data", lidar["next"])
        n += 1
        if n % frame_skip == 0:
            tokens.append(lidar["token"])
    return tokens


def nearby_cameras(nusc, lidar_token: str, camera: str, max_translation: float) -> list:
    """Camera sample_data tokens near a LiDAR frame: the camera frame of the same sample, then the following frames
    and then the preceding ones, each run stopping at the first frame more than max_translation metres away.

    As in the protocol this reproduces, each run starts two frames away: the frames right next to the sample's own
    camera frame are never candidates."""
    lidar = nusc.get("sample_data", lidar_token)
    world_to_lidar_ego = np.linalg.inv(ego_pose(nusc, lidar).astype(np.float64))
    first = nusc.get("sample_data", nusc.get("sample", lidar["sample_token"])["data"][camera])
    tokens = [first["token"]]
    for direction in ("next", "prev"):
        if not first[direction]:
            continue
        cam = nusc.get("sample_data", first[direction])
        while cam[direction]:
            cam = nusc.get("sample_data", cam[direction])
            if np.linalg.norm((world_to_lidar_ego @ ego_pose(nusc, cam))[0:3, 3]) >= max_translation:
                break
            tokens.append(cam["token"])
    return tokens


def build_index(nusc, camera: str, frame_skip: int, max_translation: float) -> list:
    """[(lidar_token, [nearby camera tokens])] over the daytime scenes, in scene order."""
    lidar_tokens = [t for s in daytime_scenes(nusc) for t in lidar_frames(nusc, s, frame_skip)]
    return [(t, nearby_cameras(nusc, t, camera, max_translation)) for t in tqdm(lidar_tokens, desc="index")]


# ---------------------------------------------------------------------------
# 2. Samples
# ---------------------------------------------------------------------------

def load_sweep(nusc, token: str):
    """(3 x N points, 1 x N intensity) in the LiDAR frame with points on the ego car removed, and the sweep's ego
    pose."""
    from nuscenes.utils.data_classes import LidarPointCloud

    sd = nusc.get("sample_data", token)
    pts = LidarPointCloud.from_file(os.path.join(nusc.dataroot, sd["filename"])).points
    on_car = (np.abs(pts[0]) < EGO_CAR_HALF_EXTENT[0]) & (np.abs(pts[1]) < EGO_CAR_HALF_EXTENT[1])
    return pts[0:3, ~on_car], pts[3:4, ~on_car], ego_pose(nusc, sd)


def accumulate(nusc, lidar, accum_frames: int, accum_skip: int):
    """The sweep plus up to accum_frames sweeps after it and before it, taking every accum_skip-th, in its LiDAR
    frame. Returns (3 x N points, 1 x N intensity)."""
    pts, inten, ego_i = load_sweep(nusc, lidar["token"])
    pts_list, inten_list = [pts], [inten]
    world_to_ego_i = np.linalg.inv(ego_i)
    lidar_to_ego = sensor_to_ego(nusc, lidar)
    ego_to_lidar = np.linalg.inv(lidar_to_ego)
    for direction in ("next", "prev"):
        sd, n, taken = lidar, 1, 0
        while taken < accum_frames and sd[direction]:
            if n % accum_skip == 0:
                pts_j, inten_j, ego_j = load_sweep(nusc, sd[direction])
                T = ego_to_lidar @ (world_to_ego_i @ ego_j) @ lidar_to_ego
                pts_list.append((T @ np.concatenate((pts_j, np.ones((1, pts_j.shape[1]), pts_j.dtype))))[0:3])
                inten_list.append(inten_j)
                taken += 1
            n += 1
            sd = nusc.get("sample_data", sd[direction])
    return np.concatenate(pts_list, axis=1), np.concatenate(inten_list, axis=1)


def make_sample(nusc, lidar_token: str, camera_token: str, accum: tuple, min_in_image: int, cloud=None):
    """The sample for one LiDAR frame seen from one camera frame, or None when too few points land in the image.
    cloud: the accumulated (points, intensity), when the caller already has it."""
    lidar = nusc.get("sample_data", lidar_token)
    pts, inten = cloud if cloud is not None else accumulate(nusc, lidar, *accum)
    pts, inten = pts.astype(np.float32), inten.astype(np.float32)
    cam = nusc.get("sample_data", camera_token)

    img = np.array(Image.open(os.path.join(nusc.dataroot, cam["filename"])))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    K = np.asarray(nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])["camera_intrinsic"], np.float32)
    cam_to_ego, cam_ego = sensor_to_ego(nusc, cam), ego_pose(nusc, cam)
    lidar_to_cam = np.linalg.inv(cam_to_ego) @ (np.linalg.inv(cam_ego) @ (ego_pose(nusc, lidar) @ sensor_to_ego(nusc, lidar)))
    pts_cam = lidar_to_cam[0:3, 0:3] @ pts + lidar_to_cam[0:3, 3:]

    H, W = img.shape[:2]
    uvz = K @ pts_cam
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvz[0:2] / uvz[2:]
    in_image = (uvz[2] > 0) & (uv[0] >= 0) & (uv[0] <= W - 1) & (uv[1] >= 0) & (uv[1] <= H - 1)
    if int(in_image.sum()) <= min_in_image:
        return None
    return {"img": img, "K": K, "PC": np.concatenate((pts_cam, inten), axis=0).astype(np.float32),
            "T_world_cam": (cam_ego @ cam_to_ego).astype(np.float32), "P": lidar_to_cam.astype(np.float32)}


def save_sample(out: Path, i: int, sample: dict):
    stem = f"{i:06d}"
    cv2.imwrite(str(out / "img" / f"{stem}.png"), sample["img"])
    for key in ("K", "PC", "T_world_cam", "P"):
        np.save(out / key / f"{stem}.npy", sample[key])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", required=True, help="nuScenes root (samples/, sweeps/, <version>/)")
    p.add_argument("--version", default="v1.0-trainval", help="v1.0-trainval for training, v1.0-test for testing")
    p.add_argument("--out", required=True, help="Output folder; gets img/, K/, PC/, T_world_cam/ and P/")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--frame_skip", type=int, default=10,
                   help="Keep every n-th LiDAR sweep: 10 built the training set, 2 the CorrI2P test set")
    p.add_argument("--max_translation", type=float, default=10.0, help="Metres between LiDAR and camera ego poses")
    p.add_argument("--accum_frames", type=int, default=3, help="Sweeps accumulated on each side")
    p.add_argument("--accum_skip", type=int, default=4, help="Take every n-th sweep when accumulating")
    p.add_argument("--min_points", type=int, default=45000)
    p.add_argument("--min_in_image", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0, help="Seeds the camera frame picked for each LiDAR frame")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many LiDAR frames (for a quick check)")
    args = p.parse_args()

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    index = build_index(nusc, args.camera, args.frame_skip, args.max_translation)[:args.limit]
    out = Path(args.out)
    for key in ("img", "K", "PC", "T_world_cam", "P"):
        (out / key).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n_kept = 0
    for lidar_token, cameras in tqdm(index, desc="samples"):
        cloud = accumulate(nusc, nusc.get("sample_data", lidar_token), args.accum_frames, args.accum_skip)
        if cloud[0].shape[1] < args.min_points or not cameras:
            continue
        sample = make_sample(nusc, lidar_token, rng.choice(cameras), (args.accum_frames, args.accum_skip),
                             args.min_in_image, cloud=cloud)
        if sample is not None:
            save_sample(out, n_kept, sample)
            n_kept += 1
    print(f"{n_kept} of {len(index)} LiDAR frames kept")


if __name__ == "__main__":
    main()
