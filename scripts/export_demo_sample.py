#!/usr/bin/env python3
"""Export one query frame + map cloud from a benchmark dataset into the demo sample format.

    <out>/image.<ext>  query image (copied unchanged; osd: undistorted to pinhole)
    <out>/K.npy        3x3 pinhole intrinsics of the image (float32)
    <out>/cloud.npy    (N, 3) float32 map cloud in metres, exactly as fed to the model
    <out>/meta.json    grid_size, title, source, and ground truth when available
                       (T_c2w: 4x4 camera-to-world pose in the cloud frame)

The result runs with:  python scripts/demo.py --sample <out>

With --stream (polycam only) it writes a stream sample for scripts/demo_localization.py instead:

    <out>/cloud.npy        the map, as above
    <out>/frames/NNN.jpg   query images (copied unchanged)
    <out>/meta.json        grid_size, title, source, and "frames": [{"image", "K", "T_c2w"}, ...]
This is how the Hub samples (fulkast/drs_vpt_demo_data) were made. Examples:

    python scripts/export_demo_sample.py --dataset kitti --data /path/to/kitti_vp2p_eval_data --sample 539 --out kitti
    python scripts/export_demo_sample.py --dataset kitti_raw --data /path/to/2011_09_26/2011_09_26_drive_0005_sync --sample 50 --out kitti_raw
    python scripts/export_demo_sample.py --dataset rio10 --data /path/to/rio10 --out rio10
    python scripts/export_demo_sample.py --dataset polycam --data /path/to/polycam_export --out my_room
    python scripts/export_demo_sample.py --dataset osd --data /path/to/osd-dense-depth --scene keble-01 --out keble
    python scripts/export_demo_sample.py --dataset polycam --stream --data /path/to/polycam_export --out my_room_stream
    python scripts/export_demo_sample.py --dataset custom --image img.jpg --cloud map.npy --fx 500 --fy 500 --cx 320 --cy 240 --out mine
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np


def _centered(raw_cloud: np.ndarray):
    centroid = raw_cloud.mean(0)
    return raw_cloud - centroid, centroid


def _c2w_in_cloud_frame(pose_txt: Path, centroid: np.ndarray):
    """Read a 4x4 camera-to-world pose and shift it into the centroid-subtracted cloud frame."""
    try:
        T = np.loadtxt(pose_txt).reshape(4, 4)
    except (OSError, ValueError):
        return None
    if not np.isfinite(T).all():  # e.g. 12-Scenes writes "-1.#INF" where tracking failed
        return None
    T[:3, 3] -= centroid
    return T.tolist()


def _ply_vertices(path: Path) -> np.ndarray:
    import trimesh
    return np.array(trimesh.load(str(path)).vertices, dtype=np.float32)


# ---------------------------------------------------------------------------
# Datasets. Each returns (img_path, K, cloud, grid_size, meta).
# ---------------------------------------------------------------------------

def kitti(args):
    from drs_vpt.datasets.kitti_vp2p import kitti_pc_img_dataset

    idx = 539 if args.sample is None else args.sample
    dataset = kitti_pc_img_dataset(args.data, "val", 1_000_000)
    # The eval set applies a random rigid transform and point shuffle per sample; seed for repeatability.
    random.seed(0)
    np.random.seed(0)
    data = dataset[idx]
    cloud = data["pc"].T.numpy()[:, :3].astype(np.float32)
    return data["img_path"], data["K"].numpy(), cloud, 0.02, {
        "title": f"KITTI sample {idx}", "source": "KITTI odometry (VP2P eval split, 'val')",
        "sample_index": idx, "P": data["P"].numpy().tolist(),
        "P_note": "'P' exactly as returned by kitti_pc_img_dataset for this sample",
    }


def _kitti_calib_file(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        k, _, v = line.partition(":")
        try:
            out[k] = np.array([float(x) for x in v.split()])
        except ValueError:
            pass  # calib_time
    return out


def kitti_raw_frame(drive: Path, idx: int):
    """One frame of a KITTI raw synced+rectified drive: image_02, its rectified K, the raw Velodyne sweep (LiDAR frame)
    and the factory T_cam2_velo (x_cam2rect = T @ x_velo) from the date folder's calib files."""
    drive = Path(drive)
    c2c = _kitti_calib_file(drive.parent / "calib_cam_to_cam.txt")
    v2c = _kitti_calib_file(drive.parent / "calib_velo_to_cam.txt")
    P = c2c["P_rect_02"].reshape(3, 4)
    K = P[:, :3]
    T_c0_velo = np.eye(4)
    T_c0_velo[:3, :3], T_c0_velo[:3, 3] = v2c["R"].reshape(3, 3), v2c["T"]
    R_rect = np.eye(4)
    R_rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    T_c2_c0 = np.eye(4)
    T_c2_c0[:3, 3] = np.linalg.solve(K, P[:, 3])  # rectified cam 2 sits at this offset from rectified cam 0
    sweep = np.fromfile(drive / "velodyne_points/data" / f"{idx:010d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
    return drive / "image_02/data" / f"{idx:010d}.png", K, sweep, T_c2_c0 @ R_rect @ T_c0_velo


def kitti_raw(args):
    """KITTI raw (--data: a <date>/<drive>_sync folder): image_02 frame --sample against its own Velodyne sweep.

    The cloud is the single raw sweep in the LiDAR frame (centroid-subtracted), so the ground-truth pose is the factory
    camera-LiDAR calibration and a localization is a calibration estimate.
    """
    idx = 0 if args.sample is None else args.sample
    img, K, sweep, T_c_velo = kitti_raw_frame(Path(args.data), idx)
    cloud, centroid = _centered(sweep.astype(np.float64))
    T = np.linalg.inv(T_c_velo)
    T[:3, 3] -= centroid
    drive = Path(args.data).resolve()
    return img, K, cloud, args.grid_size or 0.02, {
        "title": f"KITTI raw {drive.name} frame {idx}",
        "source": f"KITTI raw {drive.parent.name}/{drive.name}, image_02 + velodyne frame {idx}; map = that single "
                  "sweep, centroid-subtracted; T_c2w = factory calibration (calib_velo_to_cam + rectification)",
        "T_c2w": T.tolist(),
        "license": "CC BY-NC-SA 3.0",
        "attribution": "Geiger et al., Vision meets Robotics: The KITTI Dataset, IJRR 2013",
    }


def boreas_frame(seq: Path, idx: int):
    """Camera frame idx (index into the local camera/*.png) of a Boreas sequence, the LiDAR scan nearest in time (LiDAR
    frame) and the published T_camera_lidar. Images are rectified; K is P_camera. Also returns the time from the scan's
    points in front of the camera to the image, in seconds (scan columns: x y z intensity ring t_rel)."""
    seq = Path(seq)
    img = sorted((seq / "camera").glob("*.png"))[idx]
    scans = sorted((seq / "lidar").glob("*.bin"))
    t_img = int(img.stem) * 1e-6
    scan = min(scans, key=lambda s: abs(int(s.stem) * 1e-6 - t_img))
    pts = np.fromfile(scan, dtype=np.float32).reshape(-1, 6)
    T = np.loadtxt(seq / "calib/T_camera_lidar.txt")
    K = np.loadtxt(seq / "calib/P_camera.txt")[:3, :3]
    z = pts[:, :3] @ T[2, :3] + T[2, 3]
    ahead = (z > 2) & (np.abs(pts[:, :3] @ T[0, :3] + T[0, 3]) < 0.2 * z)  # within ~11 deg of the optical axis
    dt = t_img - (int(scan.stem) * 1e-6 + float(np.median(pts[ahead, 5])))
    return img, K, pts[:, :3], T, dt


def boreas(args):
    """Boreas (--data: a local boreas-<date> folder with camera/, lidar/, calib/ as in s3://boreas): camera frame
    --sample against the nearest single LiDAR scan (LiDAR frame, centroid-subtracted). The ground-truth pose is the
    published camera-LiDAR calibration, so a localization is a calibration estimate."""
    idx = 0 if args.sample is None else args.sample
    img, K, scan, T_c_lidar, dt = boreas_frame(Path(args.data), idx)
    cloud, centroid = _centered(scan.astype(np.float64))
    T = np.linalg.inv(T_c_lidar)
    T[:3, 3] -= centroid
    seq = Path(args.data).resolve().name
    return img, K, cloud, args.grid_size or 0.02, {
        "title": f"Boreas {seq} camera {img.stem}",
        "source": f"Boreas {seq}, camera/{img.name} + nearest lidar scan ({dt * 1e3:+.0f} ms to the points ahead); "
                  "map = that single scan, centroid-subtracted; T_c2w = published calib/T_camera_lidar",
        "T_c2w": T.tolist(),
        "attribution": "Burnett et al., Boreas: A Multi-Season Autonomous Driving Dataset, IJRR 2023",
    }


def nuscenes(args):
    from drs_vpt.datasets.utils import apply_transform_from_left

    root = Path(args.data)
    names = sorted({x.split(".")[0] for x in os.listdir(root / "img") if "clou" not in x})
    idx = 0 if args.sample is None else args.sample
    name = names[idx]
    raw = np.load(root / "PC" / f"{name}.npy").T.astype(np.float32)[:, :3]
    cloud = apply_transform_from_left(np.load(root / "P" / f"{name}.npy"), raw)
    return root / "img" / f"{name}.png", np.load(root / "K" / f"{name}.npy"), cloud, 0.02, {
        "title": f"NuScenes sample {idx}", "source": f"nuScenes (preprocessed test set) {name}",
    }


def seven_scenes(args):
    scene, seq, frame = args.scene or "stairs", "seq-01", "frame-000000"
    base = Path(args.data) / scene
    raw = np.loadtxt(base / "tsdf_points_txt" / f"{scene}_train_tsdf_points.txt").astype(np.float32)
    cloud, centroid = _centered(raw)
    K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float32)
    return base / seq / f"{frame}.color.png", K, cloud, 0.01, {
        "title": f"7Scenes {scene}/{seq}/{frame}",
        "source": f"7-Scenes {scene}/{seq}/{frame}; map = TSDF points fused from the train split, centroid-subtracted",
        "centroid": centroid.tolist(), "T_c2w": _c2w_in_cloud_frame(base / seq / f"{frame}.pose.txt", centroid),
    }


def twelve_scenes(args):
    import re

    scene = args.scene or "apt2/living"
    base = Path(args.data) / scene
    # Test frames are sequence0 in split.txt
    m = re.search(r"sequence0\s*\[frames=\d+\]\s*\[start=(\d+)\s*;\s*end=(\d+)\]", (base / "split.txt").read_text())
    frame = f"frame-{list(range(int(m.group(1)), int(m.group(2))))[args.sample or 0]:06d}"
    ply = next(f for f in os.listdir(base) if f.endswith(".ply"))
    cloud, centroid = _centered(_ply_vertices(base / ply))
    K = np.array([[1158.3, 0, 649.0], [0, 1153.53, 483.5], [0, 0, 1.0]], dtype=np.float32)
    return base / "data" / f"{frame}.color.jpg", K, cloud, 0.005, {
        "title": f"12Scenes {scene} {frame}",
        "source": f"12-Scenes {scene} {frame}; map = {ply} vertices, centroid-subtracted",
        "centroid": centroid.tolist(), "T_c2w": _c2w_in_cloud_frame(base / "data" / f"{frame}.pose.txt", centroid),
    }


def _cambridge_points3d(bin_path: Path, min_track_len: int = 3) -> np.ndarray:
    """COLMAP points3D.bin, keeping points seen in >= min_track_len images (fewer are mostly outliers)."""
    import struct
    xyzs = []
    with open(bin_path, "rb") as f:
        (num_pts,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_pts):
            blob = f.read(43)              # point3D_id(8) + xyz(24) + rgb(3) + error(8)
            x, y, z = struct.unpack_from("<ddd", blob, 8)
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * track_len, 1)       # skip (image_id, point2D_idx) pairs
            if track_len >= min_track_len:
                xyzs.append((x, y, z))
    return np.array(xyzs, dtype=np.float32)


def _cambridge_intrinsics(colmap_model: Path, query_name: str) -> np.ndarray:
    """3x3 K for a query image from the COLMAP empty_all text model (SIMPLE_RADIAL: f cx cy k1)."""
    name_to_cam = {}
    for line in (colmap_model / "empty_all" / "images.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 10 and not line.startswith("#"):
            try:
                name_to_cam[parts[9]] = int(parts[8])
            except ValueError:
                pass
    cam_id = name_to_cam[query_name]
    for line in (colmap_model / "empty_all" / "cameras.txt").read_text().splitlines():
        parts = line.split()
        if parts and not line.startswith("#") and int(parts[0]) == cam_id:
            f, cx, cy = map(float, parts[4:7])
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    raise ValueError(f"camera_id {cam_id} not found for query {query_name}")


def cambridge(args):
    base = Path(args.data)
    scene = args.scene or "KingsCollege"
    colmap_model = (base / "CambridgeLandmarks_Colmap_Retriangulated_1024px" /
                    "CambridgeLandmarks_Colmap_Retriangulated_1024px" / scene)
    query = args.query
    if not query:
        query_list = base / scene / "dataset_test.txt"
        if not query_list.exists():
            query_list = colmap_model / "list_query.txt"
        query = next(line for line in query_list.read_text().splitlines() if "/" in line).split()[0]

    if args.dense_ply:  # dense MVS cloud, if you have one
        cloud, _ = _centered(_ply_vertices(Path(args.dense_ply)))
    else:  # sparse SfM points, minus the farthest 2% (outliers)
        cloud, _ = _centered(_cambridge_points3d(colmap_model / "model_train" / "points3D.bin"))
        dists = np.linalg.norm(cloud, axis=1)
        cloud = cloud[dists <= np.percentile(dists, 98)]
    return base / scene / query, _cambridge_intrinsics(colmap_model, query), cloud, args.grid_size or 0.02, {
        "title": f"Cambridge {scene} ({query})", "source": f"Cambridge Landmarks {scene} {query}",
    }


def rio10(args):
    import yaml

    scene, ref_seq, query_seq, frame = args.scene or "scene01", "seq01_01", "seq01_02", "frame-000100"
    base = Path(args.data) / scene
    fx, fy, cx, cy = yaml.safe_load((base / "seq01" / query_seq / "camera.yaml").read_text())["camera_intrinsics"]["model"]
    cloud, centroid = _centered(_ply_vertices(base / "models01" / ref_seq / "labels.ply"))
    return base / "seq01" / query_seq / f"{frame}.color.jpg", \
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32), cloud, 0.01, {
            "title": f"RIO10 {scene}/{query_seq}/{frame}",
            "source": f"RIO10 {scene}: query {query_seq}/{frame}, map = models01/{ref_seq}/labels.ply, centroid-subtracted",
            "centroid": centroid.tolist(),
            "T_c2w": _c2w_in_cloud_frame(base / "seq01" / query_seq / f"{frame}.pose.txt", centroid),
        }


def _polycam_keyframes(data: str):
    """Keyframe ids and camera JSONs of a Polycam raw export (iPhone LiDAR)."""
    root = Path(data) / "keyframes"
    ids = sorted(p.stem for p in (root / "cameras").glob("*.json"))
    cams = {i: json.loads((root / "cameras" / f"{i}.json").read_text()) for i in ids}
    return root, ids, cams


def _polycam_c2w(c):  # ARKit camera axes (x right, y up, z backward) -> OpenCV (x right, y down, z forward)
    T = np.eye(4)
    T[:3] = [[c[f"t_{r}{k}"] for k in range(4)] for r in range(3)]
    return T @ np.diag([1.0, -1.0, -1.0, 1.0])


def _polycam_K(c, scale=1.0):
    return np.array([[c["fx"] * scale, 0, c["cx"] * scale], [0, c["fy"] * scale, c["cy"] * scale], [0, 0, 1]])


def _polycam_cloud(root: Path, ids: list, cams: dict, keep: list):
    """Back-project the high-confidence LiDAR depth of the keyframes `keep` (indices into ids) into the world
    frame, then centre it. Returns (cloud, centroid)."""
    from PIL import Image

    clouds = []
    for k in keep:
        i = ids[k]
        depth = np.array(Image.open(root / "depth" / f"{i}.png")).astype(np.float32) / 1000.0  # mm -> m
        conf = np.array(Image.open(root / "confidence" / f"{i}.png"))
        K = _polycam_K(cams[i], depth.shape[1] / cams[i]["width"])
        v, u = np.nonzero((conf == 255) & (depth > 0))  # 255 = ARKit high confidence
        z = depth[v, u]
        pts = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
        T = _polycam_c2w(cams[i])
        clouds.append(pts @ T[:3, :3].T + T[:3, 3])
    return _centered(np.concatenate(clouds).astype(np.float32))


def polycam(args):
    """Polycam raw export (iPhone LiDAR): keyframes/{images,depth,confidence,cameras}.

    One keyframe is the query; the map is the back-projected high-confidence LiDAR depth of the other
    keyframes, excluding the query's temporal neighbours so the map never sees the query's own depth.
    """
    root, ids, cams = _polycam_keyframes(args.data)
    q = len(ids) // 2 if args.sample is None else args.sample
    keep = [k for k in range(len(ids)) if abs(k - q) > args.exclude_neighbours]
    cloud, centroid = _polycam_cloud(root, ids, cams, keep)

    T_q = _polycam_c2w(cams[ids[q]])
    T_q[:3, 3] -= centroid
    return root / "images" / f"{ids[q]}.jpg", _polycam_K(cams[ids[q]]), cloud, args.grid_size or 0.01, {
        "title": f"iPhone LiDAR {Path(args.data).name} keyframe {q}",
        "source": f"Polycam raw export {Path(args.data).name}: query keyframe {q} ({ids[q]}); map = high-confidence "
                  f"LiDAR depth of {len(keep)} other keyframes (excluding +/-{args.exclude_neighbours} neighbours), "
                  "centroid-subtracted",
        "centroid": centroid.tolist(), "T_c2w": T_q.tolist(),
        "up": [0.0, 1.0, 0.0],  # ARKit world is y-up; lets the viz show portrait captures upright
    }


def polycam_stream(args):
    """A stream sample from a Polycam raw export: every --stride-th keyframe is a query frame, and the map is the
    LiDAR depth of all keyframes. Each query's own depth is in the map, so GT errors are optimistic."""
    root, ids, cams = _polycam_keyframes(args.data)
    cloud, centroid = _polycam_cloud(root, ids, cams, list(range(len(ids))))
    frames = []
    for k in range(0, len(ids), args.stride):
        T = _polycam_c2w(cams[ids[k]])
        T[:3, 3] -= centroid
        frames.append({"keyframe": k, "src": root / "images" / f"{ids[k]}.jpg",
                       "K": _polycam_K(cams[ids[k]]).tolist(), "T_c2w": T.tolist()})
    return frames, cloud, args.grid_size or 0.01, {
        "title": f"iPhone LiDAR {Path(args.data).name} stream",
        "source": f"Polycam raw export {Path(args.data).name}: every {args.stride} of {len(ids)} keyframes as query "
                  f"frames; map = high-confidence LiDAR depth of all {len(ids)} keyframes (includes each query's own "
                  "depth), centroid-subtracted",
        "centroid": centroid.tolist(),
        "up": [0.0, 1.0, 0.0],
    }


def _osd_visible_points(tiles, cam, max_range, scale=0.5, tol=0.01, occl_k=5):
    """TLS points visible from one fisheye camera: z-buffer on a (scale x) pixel grid, keeping every point within
    tol (relative) of the nearest one at its pixel, after osd-dense-depth's see-through filter."""
    import dataclasses

    import cv2
    from osd_dense_depth.camera import project

    c = dataclasses.replace(cam, w=int(cam.w * scale), h=int(cam.h * scale),
                            K=np.diag([scale, scale, 1.0]) @ cam.K)
    T_C_G = np.linalg.inv(cam.T_G_C)
    R, t = T_C_G[:3, :3].astype(np.float32), T_C_G[:3, 3].astype(np.float32)

    def in_view(P):
        Pc = P @ R.T + t
        r = np.linalg.norm(Pc, axis=1)
        pre = np.flatnonzero((r < max_range) & (Pc[:, 2] > 0.05))
        uv, ok = project(Pc[pre], c)
        idx = pre[ok]
        pix = (np.clip(np.rint(uv[ok, 1]), 0, c.h - 1).astype(np.int64) * c.w
               + np.clip(np.rint(uv[ok, 0]), 0, c.w - 1).astype(np.int64))
        return idx, pix, r[idx]

    names = tiles.in_view(T_C_G, max_range, 80.0)
    best = np.full(c.w * c.h, np.inf, np.float32)
    for name in names:
        _, pix, r = in_view(tiles.get(name))
        np.minimum.at(best, pix, r.astype(np.float32))
    valid = np.isfinite(best).reshape(c.h, c.w)
    zmin = cv2.erode(np.where(valid, best.reshape(c.h, c.w), 1e6).astype(np.float32), np.ones((occl_k, occl_k), np.uint8))
    best[~(valid & ~(best.reshape(c.h, c.w) > 1.1 * zmin + 0.2)).ravel()] = -1   # see-through pixels keep nothing
    out = []
    for name in names:
        P = tiles.get(name)
        idx, pix, r = in_view(P)
        out.append(P[idx[r <= best[pix] * (1 + tol)]])
    return np.concatenate(out) if out else np.zeros((0, 3), np.float32)


def osd(args):
    """Oxford Spires via osd-dense-depth (--data: that repo): refined camera poses + the site's TLS map.

    The query is one image of --osd-camera, undistorted from the fisheye model to a pinhole with the same fx, fy, cx,
    cy. The map is the TLS points visible from all three cameras of every rig frame within --osd-window seconds of
    the query, up to --osd-range metres, voxelized at --grid-size. The TLS map was scanned independently of the
    robot run, so the query's own view is part of the map without leaking anything about the query image.
    """
    import sys

    import cv2

    sys.path.insert(0, str(Path(args.data).resolve()))
    from osd_dense_depth.camera import read_cameras
    from osd_dense_depth.download import fetch_cloud, fetch_images
    from osd_dense_depth.sequences import CLOUDS, SEQUENCES, cameras_csv
    from osd_dense_depth.tiles import Tiles, build_tiles

    seq = args.scene or "keble-01"
    seq_hf, site = SEQUENCES[seq]
    work = Path(args.data) / "osd_depth"
    tile_dir = work / "_tiles" / site
    if not (tile_dir / "index.json").exists():
        print(f"Cutting the {site} TLS map into tiles -> {tile_dir} (one-off)")
        build_tiles(fetch_cloud(CLOUDS[site]), tile_dir)
    tiles = Tiles(tile_dir)

    cams = read_cameras(cameras_csv(seq))
    stamp = [float(Path(c.image).stem) for c in cams]
    which = [i for i, c in enumerate(cams) if c.camera == f"cam_{args.osd_camera}"]
    q = which[len(which) // 2 if args.sample is None else args.sample]
    qc = cams[q]
    near = [i for i in range(len(cams)) if abs(stamp[i] - stamp[q]) <= args.osd_window]

    pts = np.concatenate([_osd_visible_points(tiles, cams[i], args.osd_range) for i in near])
    grid = args.grid_size or 0.02
    _, keep = np.unique(np.floor(pts / grid).astype(np.int64), axis=0, return_index=True)
    cloud, centroid = _centered(pts[keep].astype(np.float64))

    fetch_images(seq_hf, [qc.image], work / seq)
    img = cv2.imread(str(work / seq / qc.image))
    K = qc.K.copy()
    maps = cv2.fisheye.initUndistortRectifyMap(qc.K, qc.D, np.eye(3), K, (qc.w, qc.h), cv2.CV_32FC1)
    pin = work / seq / "pinhole" / qc.image
    pin.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pin), cv2.remap(img, *maps, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])

    T = qc.T_G_C.copy()
    T[:3, 3] -= centroid
    return pin, K, cloud, grid, {
        "title": f"Oxford Spires {seq}: {args.osd_camera} camera {Path(qc.image).stem}",
        "source": f"Oxford Spires Dataset {seq_hf}, {qc.image} (undistorted to pinhole); map = TLS points "
                  f"({CLOUDS[site]}) visible from the {len(near)} cameras within {args.osd_window:g} s, up to "
                  f"{args.osd_range:g} m, {grid:g} m voxels, centroid-subtracted; pose from osd-dense-depth",
        "T_c2w": T.tolist(),
        "license": "CC BY-NC-SA 4.0",
        "attribution": "Oxford Spires Dataset (Tao et al., IJRR 2025), Dynamic Robot Systems group, "
                       "Oxford Robotics Institute",
    }


def custom(args):
    if not args.image or not args.cloud:
        raise SystemExit("--dataset custom needs --image and --cloud")
    path = Path(args.cloud)
    cloud = (np.load(path) if path.suffix == ".npy" else np.loadtxt(path)).astype(np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise SystemExit(f"Cloud must be (N, >=3), got {cloud.shape}")
    K = np.array([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]], dtype=np.float32)
    return args.image, K, cloud[:, :3], args.grid_size or 0.02, {
        "title": f"Custom: {Path(args.image).name}", "source": f"{args.image} + {args.cloud}",
    }


DATASETS = {"kitti": kitti, "kitti_raw": kitti_raw, "boreas": boreas, "nuscenes": nuscenes,
            "7scenes": seven_scenes, "12scenes": twelve_scenes, "cambridge": cambridge, "rio10": rio10,
            "polycam": polycam, "osd": osd, "custom": custom}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--data", help="Dataset root directory (all datasets except custom)")
    p.add_argument("--out", required=True, help="Output sample directory")
    p.add_argument("--sample", type=int, help="Sample/frame index (kitti, kitti_raw, boreas, nuscenes, 12scenes; polycam: query keyframe; "
                   "osd: front-camera frame, default the middle one)")
    p.add_argument("--exclude-neighbours", type=int, default=2,
                   help="Keyframes on each side of the query left out of the map (polycam, default 2)")
    p.add_argument("--scene", help="Scene name (7scenes, 12scenes, cambridge, rio10; osd: sequence, default keble-01)")
    p.add_argument("--osd-window", type=float, default=5.0,
                   help="Cameras within this many seconds of the query build the map (osd, default 5)")
    p.add_argument("--osd-camera", choices=["front", "left", "right"], default="front",
                   help="Rig camera of the query (osd, default front); --sample indexes this camera's frames")
    p.add_argument("--osd-range", type=float, default=20.0, help="Map range from each camera in metres (osd, default 20)")
    p.add_argument("--query", help="Query image relative to the scene root (cambridge)")
    p.add_argument("--dense-ply", help="Dense MVS cloud to use as the map (cambridge)")
    p.add_argument("--grid-size", type=float, help="Voxel size in metres (cambridge, custom: 0.02; polycam: 0.01)")
    p.add_argument("--stream", action="store_true",
                   help="Write a stream sample (polycam): many query frames against one map")
    p.add_argument("--stride", type=int, default=1, help="Use every stride-th keyframe as a query frame (--stream)")
    p.add_argument("--license", help="Licence of the data, written to meta.json")
    p.add_argument("--attribution", help="Attribution of the data, written to meta.json")
    p.add_argument("--image", help="Query image (custom)")
    p.add_argument("--cloud", help="Map cloud .npy or .txt, (N, >=3) (custom)")
    for k, v in {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}.items():
        p.add_argument(f"--{k}", type=float, default=v, help=f"Intrinsics {k} (custom)")
    args = p.parse_args()
    if args.dataset != "custom" and not args.data:
        p.error("--data is required")
    if args.stream and args.dataset != "polycam":
        p.error("--stream is only implemented for polycam")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stream:
        frames, cloud, grid_size, meta = polycam_stream(args)
        (out / "frames").mkdir(exist_ok=True)
        for n, f in enumerate(frames):
            src = f.pop("src")
            frames[n] = {"image": f"frames/{f['keyframe']:03d}{src.suffix}", **f}
            shutil.copy(src, out / frames[n]["image"])
        meta["frames"] = frames
    else:
        img_path, K, cloud, grid_size, meta = DATASETS[args.dataset](args)
        shutil.copy(img_path, out / f"image{Path(img_path).suffix}")
        np.save(out / "K.npy", np.asarray(K, dtype=np.float32))

    np.save(out / "cloud.npy", np.ascontiguousarray(cloud, dtype=np.float32))
    up = meta.pop("up", None)
    credits = {k: v for k, v in (("license", args.license), ("attribution", args.attribution)) if v}
    meta = {"grid_size": grid_size, **meta, **credits, **({"up": up} if up is not None else {})}  # Hub key order
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}: {len(cloud)} points, grid {grid_size} m"
          + (f", {len(meta['frames'])} frames" if args.stream else ""))


if __name__ == "__main__":
    main()
