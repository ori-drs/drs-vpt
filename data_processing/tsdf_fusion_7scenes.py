# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

import os
import re
import glob
import argparse
import numpy as np
import open3d as o3d

# ---- 7-Scenes defaults (KinectFusion intrinsics commonly used for the dataset) ----
WIDTH, HEIGHT = 640, 480
FX, FY = 585.0, 585.0
CX, CY = 320.0, 240.0


def find_train_split_file(scene_dir: str) -> str:
    candidates = [
        os.path.join(scene_dir, "TrainSplit.txt"),
        os.path.join(scene_dir, "train_split.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    globs = glob.glob(os.path.join(scene_dir, "*TrainSplit*.txt")) + glob.glob(
        os.path.join(scene_dir, "*train*split*.txt")
    )
    if globs:
        return sorted(globs)[0]

    raise FileNotFoundError(f"Could not find TrainSplit.txt in {scene_dir}")

def find_test_split_file(scene_dir: str) -> str:
    candidates = [
        os.path.join(scene_dir, "TestSplit.txt"),
        os.path.join(scene_dir, "test_split.txt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    globs = glob.glob(os.path.join(scene_dir, "*TestSplit*.txt")) + glob.glob(
        os.path.join(scene_dir, "*test*split*.txt")
    )
    if globs:
        return sorted(globs)[0]

    raise FileNotFoundError(f"Could not find TestSplit.txt in {scene_dir}")


def parse_seq_ids(split_path: str) -> list[int]:
    """
    Returns all sequence ids found in TrainSplit.txt (unique, in file order).
    """
    txt = open(split_path, "r", encoding="utf-8", errors="ignore").read()
    nums = [int(x) for x in re.findall(r"-?\d+", txt)]
    if not nums:
        raise ValueError(f"No integers found in {split_path}")

    seen = set()
    seq_ids = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            seq_ids.append(n)
    return seq_ids


def read_pose_4x4(pose_path: str) -> np.ndarray:
    T = np.loadtxt(pose_path).astype(np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{pose_path} is not 4x4, got {T.shape}")
    return T


def integrate_sequence_into_volume(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    seq_dir: str,
    depth_scale: float = 1000.0,  # mm -> m
    depth_trunc: float = 3.0,     # meters
) -> int:
    """
    Integrate all frames from one seq-* directory into the provided TSDF volume.
    Returns number of integrated frames (with poses present).
    """
    depth_paths = sorted(glob.glob(os.path.join(seq_dir, "frame-*.depth.png")))
    if not depth_paths:
        raise FileNotFoundError(f"No depth frames found in {seq_dir}")

    integrated = 0
    for depth_path in depth_paths:
        stem = os.path.basename(depth_path).replace(".depth.png", "")
        pose_path = os.path.join(seq_dir, f"{stem}.pose.txt")
        if not os.path.exists(pose_path):
            continue

        Trgb_depth = np.array([[1.0, 0.0, 0.0, -0.0254],
                               [0.0, 1.0, 0.0, -0.00013],
                               [0.0, 0.0, 1.0, -0.00218],
                               [0, 0, 0, 1]])

        T_wc = read_pose_4x4(pose_path) @ Trgb_depth     # depth camera -> world

        T_cw = np.linalg.inv(T_wc)          # world -> camera

        depth = o3d.io.read_image(depth_path)

        # Dummy color image (NoColor ignores it)
        h, w = np.asarray(depth).shape[:2]
        dummy_color = o3d.geometry.Image(np.zeros((h, w, 3), dtype=np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=dummy_color,
            depth=depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        volume.integrate(rgbd, intrinsic, T_cw)
        integrated += 1

    return integrated


def fuse_all_train_seqs_to_one_txt(
    scene_dir: str,
    out_txt_path: str,
    voxel_length: float = 0.01,   # meters
    sdf_trunc: float = 0.04,      # meters
    depth_scale: float = 1000.0,  # mm -> m
    depth_trunc: float = 3.0,     # meters
) -> int:
    """
    Builds ONE TSDF volume from ALL training sequences listed in TrainSplit.txt,
    then extracts ONE point cloud and saves as Nx3 ASCII .txt.
    Returns number of points saved.
    """
    train_split = find_train_split_file(scene_dir)
    seq_ids = parse_seq_ids(train_split)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    total_frames = 0
    for seq_id in seq_ids:
        seq_name = f"seq-{seq_id:02d}"
        seq_path = os.path.join(scene_dir, seq_name)
        if not os.path.isdir(seq_path):
            print(f"[WARN] Missing {seq_name} at {seq_path}, skipping.")
            continue

        n = integrate_sequence_into_volume(
            volume=volume,
            intrinsic=intrinsic,
            seq_dir=seq_path,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
        )
        total_frames += n
        print(f"[INFO] Integrated {n} frames from {os.path.basename(scene_dir)}/{seq_name}")

    print(f"[INFO] Total integrated frames for {os.path.basename(scene_dir)}: {total_frames}")

    pcd = volume.extract_point_cloud()
    pts = np.asarray(pcd.points)

    os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)
    np.savetxt(out_txt_path, pts, fmt="%.6f")
    return int(pts.shape[0])


def fuse_all_test_seqs_to_one_txt(
    scene_dir: str,
    out_txt_path: str,
    voxel_length: float = 0.01,   # meters
    sdf_trunc: float = 0.04,      # meters
    depth_scale: float = 1000.0,  # mm -> m
    depth_trunc: float = 3.0,     # meters
) -> int:
    """
    Builds ONE TSDF volume from ALL training sequences listed in TrainSplit.txt,
    then extracts ONE point cloud and saves as Nx3 ASCII .txt.
    Returns number of points saved.
    """
    test_split = find_test_split_file(scene_dir)
    seq_ids = parse_seq_ids(test_split)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    total_frames = 0
    for seq_id in seq_ids:
        seq_name = f"seq-{seq_id:02d}"
        seq_path = os.path.join(scene_dir, seq_name)
        if not os.path.isdir(seq_path):
            print(f"[WARN] Missing {seq_name} at {seq_path}, skipping.")
            continue

        n = integrate_sequence_into_volume(
            volume=volume,
            intrinsic=intrinsic,
            seq_dir=seq_path,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
        )
        total_frames += n
        print(f"[INFO] Integrated {n} frames from {os.path.basename(scene_dir)}/{seq_name}")

    print(f"[INFO] Total integrated frames for {os.path.basename(scene_dir)}: {total_frames}")

    pcd = volume.extract_point_cloud()
    pts = np.asarray(pcd.points)

    os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)
    np.savetxt(out_txt_path, pts, fmt="%.6f")
    return int(pts.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        required=True,
        help="Path to 7-Scenes root containing scene folders",
    )
    ap.add_argument("--scene", default=None, help="Optional: only process one scene (e.g., chess)")
    ap.add_argument("--out_dir", default="tsdf_points_txt", help="Output directory (relative to each scene_dir)")
    ap.add_argument("--voxel", type=float, default=0.01, help="TSDF voxel size (m)")
    ap.add_argument("--sdf_trunc", type=float, default=0.04, help="TSDF truncation (m)")
    ap.add_argument("--depth_trunc", type=float, default=3.0, help="Depth truncation (m)")
    args = ap.parse_args()

    all_scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    if args.scene is not None:
        if args.scene not in all_scenes:
            raise ValueError(f"--scene must be one of {all_scenes}, got '{args.scene}'")
        scene_dirs = [os.path.join(args.dataset_root, args.scene)]
    else:
        scene_dirs = [os.path.join(args.dataset_root, x) for x in all_scenes]

    if not scene_dirs:
        raise FileNotFoundError("No scene folders found.")

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir.rstrip("/"))

        # Output directory is relative to scene_dir (per your note)
        out_base = os.path.join(scene_dir, args.out_dir)
        os.makedirs(out_base, exist_ok=True)

        out_txt = os.path.join(out_base, f"{scene_name}_test_tsdf_points.txt")
        print(f"[INFO] {scene_name}: fusing ALL TestSplit sequences -> {out_txt}")
        n_pts = fuse_all_test_seqs_to_one_txt(
            scene_dir=scene_dir,
            out_txt_path=out_txt,
            voxel_length=args.voxel,
            sdf_trunc=args.sdf_trunc,
            depth_trunc=args.depth_trunc,
        )
        print(f"[DONE] {scene_name}: saved {n_pts} points")

        out_txt = os.path.join(out_base, f"{scene_name}_train_tsdf_points.txt")
        print(f"[INFO] {scene_name}: fusing ALL TrainSplit sequences -> {out_txt}")

        n_pts = fuse_all_train_seqs_to_one_txt(
            scene_dir=scene_dir,
            out_txt_path=out_txt,
            voxel_length=args.voxel,
            sdf_trunc=args.sdf_trunc,
            depth_trunc=args.depth_trunc,
        )
        print(f"[DONE] {scene_name}: saved {n_pts} points")


if __name__ == "__main__":
    main()
