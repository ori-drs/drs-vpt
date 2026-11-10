# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Rebuild the GrandTour inputs of grand_tour.py from the public GrandTour release on the Hugging Face Hub.

    python -m wai_processing.scripts.conversion.grand_tour_prepare --download /raw/grand_tour_hub \\
        --out /data/grand_tour_prepared

For each mission in assets/grand_tour/ this downloads the Hesai scans and three Alphasense cameras from
leggedrobotics/grand_tour_dataset (zarr + JPEG, no registration) and writes <out>/<mission>/ with what grand_tour.py
reads:
    cameracameralidar_calibration.yaml, T_cpt7_hesai.yaml   (shipped)
    vilens/slam_poses.csv                                   (shipped: base-frame poses from VILENS SLAM)
    vilens/individual_clouds/cloud_<sec>_<nsec>.pcd         (from data/hesai_points)
    colmap/extracted_images/<camera topic>/<mission>_nuc_alphasense_color_<stamp>.png
                                                            (from images/alphasense_{front_center,left,right})

frames.csv gives, for each cloud, the Hesai sweep it comes from and the base-frame twist (v from the SLAM poses, w
from the CPT7 IMU) that motion-compensates that sweep to the cloud's stamp (the camera trigger).
"""

import argparse
import csv
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
import zarr
from scipy.spatial.transform import Rotation as R

ASSETS = Path(__file__).parents[2] / "assets" / "grand_tour"
HUB_REPO = "leggedrobotics/grand_tour_dataset"
# Release camera name -> folder name grand_tour.py expects (the calibration's cam1, cam4, cam5).
CAMERAS = {"alphasense_front_center": "gt_box_alphasense_driver_node_cam1_color_corrected_image_compressed",
           "alphasense_left": "gt_box_alphasense_driver_node_cam4_color_corrected_image_compressed",
           "alphasense_right": "gt_box_alphasense_driver_node_cam5_color_corrected_image_compressed"}
IMAGE_SLACK_NS = 3_000_000  # grand_tour.py pairs images and clouds within 3 ms
SCAN_SLACK_NS = 1_000_000   # the release stores stamps as float seconds (sub-microsecond rounding)
STAMP_ROUNDING_NS = 1_000    # float-second stamps are within a microsecond of the original nanoseconds


def fetch(mission: str, download: Path) -> Path:
    """Download and unpack the mission's Hesai scans and camera images; returns the unpacked root."""
    from huggingface_hub import hf_hub_download

    root = download / mission
    files = ["data/hesai_points.tar"] + [f"{d}/{c}.tar" for c in CAMERAS for d in ("data", "images")]
    for f in files:
        target = root / f[:-4]
        if target.exists():
            continue
        tar = hf_hub_download(HUB_REPO, f"{mission}/{f}", repo_type="dataset", local_dir=download)
        target.mkdir(parents=True)
        with tarfile.open(tar) as t:
            t.extractall(target, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    return root


def group(root: Path, kind: str, name: str):
    return zarr.open_group(str(root / kind / name / name), mode="r")


def deskew(p, t, stamp_ns, v, w, T_base_lidar):
    """Lidar-frame sweep (per-point times in s) -> base frame at stamp_ns under a constant base-frame twist (v, w)."""
    pb = p @ T_base_lidar[:3, :3].T + T_base_lidar[:3, 3]
    dt = t - stamp_ns * 1e-9
    return R.from_rotvec(w[None] * dt[:, None]).apply(pb) + v[None] * dt[:, None]


def prepare_mission(mission: str, download: Path, out: Path):
    src, dst = ASSETS / mission, out / mission
    hub = fetch(mission, download)
    (dst / "vilens" / "individual_clouds").mkdir(parents=True, exist_ok=True)
    shutil.copy(ASSETS / "cameracameralidar_calibration.yaml", dst)
    shutil.copy(src / "T_cpt7_hesai.yaml", dst)
    shutil.copy(src / "slam_poses.csv", dst / "vilens")
    T_base_lidar = np.array(yaml.safe_load(open(src / "T_cpt7_hesai.yaml")))
    frames = list(csv.DictReader(open(src / "frames.csv")))

    hesai = group(hub, "data", "hesai_points")
    scan_t = np.asarray(hesai["timestamp"][:])
    written = 0
    for r in frames:
        stamp, scan = int(r["stamp_ns"]), int(r["scan_stamp_ns"])
        i = int(np.argmin(np.abs(scan_t - scan * 1e-9)))
        if abs(scan_t[i] - scan * 1e-9) * 1e9 > SCAN_SLACK_NS:
            print(f"{mission}: no scan at {scan}, skipping cloud {stamp}")
            continue
        n = int(hesai["valid"][i, 0])
        v = np.array([float(r[k]) for k in ("vx", "vy", "vz")])
        w = np.array([float(r[k]) for k in ("wx", "wy", "wz")])
        cloud = deskew(np.asarray(hesai["points"][i, :n], np.float64), np.asarray(hesai["time"][i, :n]), stamp, v, w,
                       T_base_lidar)
        o3d.io.write_point_cloud(str(dst / "vilens" / "individual_clouds" / f"cloud_{stamp // 10**9}_{stamp % 10**9:09d}.pcd"),
                                 o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud)))
        written += 1
    print(f"{mission}: {written} of {len(frames)} clouds")

    clouds = np.array(sorted(int(r["stamp_ns"]) for r in frames))
    for cam, folder in CAMERAS.items():
        t = np.round(np.asarray(group(hub, "data", cam)["timestamp"][:]) * 1e9).astype(np.int64)
        j = np.clip(np.searchsorted(clouds, t), 1, len(clouds) - 1)
        nearest = np.where(np.abs(clouds[j] - t) < np.abs(clouds[j - 1] - t), clouds[j], clouds[j - 1])
        near = np.abs(nearest - t) <= IMAGE_SLACK_NS
        # The release stores stamps as float seconds; restore the exact stamp of images taken at a cloud's trigger.
        t = np.where(np.abs(nearest - t) <= STAMP_ROUNDING_NS, nearest, t)
        d = dst / "colmap" / "extracted_images" / folder
        d.mkdir(parents=True, exist_ok=True)
        for idx in np.flatnonzero(near):
            img = cv2.imread(str(hub / "images" / cam / cam / f"{idx:06d}.jpeg"))
            cv2.imwrite(str(d / f"{mission}_nuc_alphasense_color_{t[idx]}.png"), img)
        print(f"{mission}: {int(near.sum())} {cam} images")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", required=True, help="Where to download and unpack the release files (~9 GB per mission)")
    ap.add_argument("--out", required=True, help="Output root; becomes original_root for grand_tour.py")
    ap.add_argument("--missions", nargs="+", default=sorted(d.name for d in ASSETS.iterdir() if d.is_dir()))
    args = ap.parse_args()
    for mission in args.missions:
        prepare_mission(mission, Path(args.download), Path(args.out))


if __name__ == "__main__":
    main()
