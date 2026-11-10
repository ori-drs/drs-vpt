# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Write PandaSet's covisibility from the shipped pair list instead of computing it.

    python -m wai_processing.scripts.pandaset_covisibility --root /data/wai/pandaset

The released model's PandaSet covisibility is binary and came from a precomputed list of covisible frame pairs
(assets/pandaset_covisible_pairs.npz), not from covisibility.py. The list is keyed by frame name only
("<frame>_<camera>"), and the same names occur in every PandaSet sequence, so a pair is marked covisible in every
scene that contains both frames. This reproduces the training matrices exactly, including frames that are not
covisible with themselves.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from drs_vpt._vendor.mapanything.utils.wai.core import load_data, store_data

PAIRS = Path(__file__).parents[1] / "assets" / "pandaset_covisible_pairs.npz"
OUT_PATH = "covisibility/v0"


def covisibility_matrix(frame_names: list, names: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    idx = {n: i for i, n in enumerate(frame_names)}
    C = np.zeros((len(frame_names), len(frame_names)), dtype=np.uint8)
    for a, b in pairs:
        i, j = idx.get(names[a]), idx.get(names[b])
        if i is not None and j is not None:
            C[i, j] = C[j, i] = 1
    return C


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="WAI PandaSet root (one folder per scene)")
    args = p.parse_args()

    z = np.load(PAIRS)
    names, pairs = z["frame_names"], z["pairs"]
    scenes = sorted(d for d in Path(args.root).iterdir() if (d / "scene_meta.json").exists())
    for scene_root in tqdm(scenes):
        scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")
        C = covisibility_matrix([f["frame_name"] for f in scene_meta["frames"]], names, pairs)
        mmap_store_name = store_data(scene_root / OUT_PATH / "pairwise_covisibility.npy", C, "mmap")
        scene_meta["scene_modalities"]["pairwise_covisibility"] = {
            "scene_key": f"{OUT_PATH}/{mmap_store_name}", "format": "mmap"}
        store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    main()
