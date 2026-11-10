#!/usr/bin/env python3
"""Write a one-frame sample (image, K.npy, cloud.npy, meta.json) from a MegaDepth scene in WAI format.

The map is the SfM depth of the frames most covisible with the query, fused in the world frame as the MegaDepth
training loader builds it; the query's own depth is not in it. The voxel size is the median 10-NN point spacing.

    python scripts/attention_explorer/make_megadepth_sample.py --scene /data/wai/megadepth/0229_0 --out samples/bull
    python scripts/attention_explorer/make_megadepth_sample.py --scene /data/wai/megadepth/0001_0 --out samples/trafalgar \\
        --query 56 --map-views 25 --rotate-seed 10

Localize MegaDepth samples with the training point cap (export_attention.py --max-points 38912).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from drs_vpt._vendor.mapanything.utils.geometry import depthmap_to_absolute_camera_coordinates
from drs_vpt._vendor.mapanything.utils.wai.core import load_data, load_frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True, help="WAI scene folder, e.g. <root>/megadepth/0229_0")
    p.add_argument("--out", required=True, help="Output sample folder")
    p.add_argument("--query", type=int, help="Query frame index (default: the most covisible frame)")
    p.add_argument("--map-views", type=int, help="Use only the N frames most covisible with the query (default: all)")
    p.add_argument("--rotate-seed", type=int,
                   help="Rotate the map frame by a fixed random rotation; some (frame, query) pairs land the model in a "
                        "wrong solution, and a different frame avoids it")
    args = p.parse_args()

    scene_root, out = Path(args.scene), Path(args.out)
    meta = load_data(str(scene_root / "scene_meta.json"), "scene_meta")
    names = [f["frame_name"] for f in meta["frames"]]
    covis = np.asarray(np.load(scene_root / meta["scene_modalities"]["pairwise_covisibility"]["scene_key"], mmap_mode="r"))
    q = args.query if args.query is not None else int(np.argmax(covis.sum(1) + covis.sum(0)))
    covis_q = (covis[q] + covis[:, q]).astype(np.float64)
    covis_q[q] = -np.inf
    map_ids = set(np.argsort(-covis_q)[:args.map_views or len(names) - 1].tolist())
    print(f"query {q}: {names[q]} of {len(names)} frames")

    clouds, spacings = [], []
    for i in sorted(map_ids):
        fr = load_frame(str(scene_root), names[i], modalities=["depth"], scene_meta=meta)
        depth = np.nan_to_num(fr["depth"].numpy().astype(np.float32), nan=0, posinf=0, neginf=0)
        pts, mask = depthmap_to_absolute_camera_coordinates(depth, fr["intrinsics"].numpy().astype(np.float32),
                                                            camera_pose=fr["extrinsics"].numpy().astype(np.float32))
        pts = pts[mask]
        if len(pts) < 100:
            continue
        pts = pts[np.random.default_rng(i).choice(len(pts), min(len(pts), 250_000), replace=False)]
        d, _ = cKDTree(pts).query(pts, k=10)
        spacings.append(np.median(d[d > 0]))
        clouds.append(pts.astype(np.float32))
    cloud = np.vstack(clouds).astype(np.float64)
    centroid = cloud.mean(0)
    cloud -= centroid

    qf = load_frame(str(scene_root), names[q], modalities=["image"], scene_meta=meta)
    T_c2w = qf["extrinsics"].numpy().astype(np.float64)
    T_c2w[:3, 3] -= centroid
    source = (f"MegaDepth scene {scene_root.name}, query {names[q]}; map = SfM depth of the {len(clouds)} images most "
              f"covisible with the query, centroid-subtracted (up to SfM scale)")
    if args.rotate_seed is not None:
        R = Rotation.random(random_state=args.rotate_seed).as_matrix()
        cloud = cloud @ R  # x_new = R^T x
        T_c2w[:3, :3] = R.T @ T_c2w[:3, :3]
        T_c2w[:3, 3] = R.T @ T_c2w[:3, 3]
        source += "; map frame rotated by a fixed random rotation"

    out.mkdir(parents=True, exist_ok=True)
    src_img = scene_root / meta["frames"][q]["file_path"]
    shutil.copy(src_img, out / f"image{src_img.suffix.lower()}")
    np.save(out / "K.npy", qf["intrinsics"].numpy().astype(np.float32))
    np.save(out / "cloud.npy", cloud.astype(np.float32))
    (out / "meta.json").write_text(json.dumps({
        "title": f"MegaDepth {scene_root.name}", "source": source, "grid_size": float(np.max(spacings)),
        "T_c2w": T_c2w.tolist(), "centroid": centroid.tolist()}, indent=1))
    print(f"{len(cloud):,} points from {len(clouds)} views, grid {np.max(spacings):.4g} -> {out}")


if __name__ == "__main__":
    main()
