#!/usr/bin/env python3
"""Write KITTI VP2P test samples, with the evaluation's seeded random scan transform, as sample folders.

    python scripts/attention_explorer/make_kitti_samples.py --data /data/kitti_vp2p_eval_data --out samples 2233 1116 5583

Each sample goes to <out>/kitti_<index>. Localize them as the benchmark does: 0.02 m voxels, no point cap.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from drs_vpt.eval import GRID_SIZE, kitti_samples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="VP2P KITTI evaluation data root")
    p.add_argument("--out", required=True, help="Folder for the sample folders")
    p.add_argument("indices", nargs="+", type=int, help="Test sample indices")
    args = p.parse_args()

    for s in kitti_samples(args.data, names=[f"{i:06d}" for i in args.indices]):
        src = Path(s["image_path"])
        d = Path(args.out) / f"kitti_{s['name']}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / f"image{src.suffix}")
        np.save(d / "K.npy", s["K"].astype(np.float32))
        np.save(d / "cloud.npy", s["cloud"])
        (d / "meta.json").write_text(json.dumps({
            "title": f"KITTI {src.parts[-3]} {src.stem}",
            "source": f"KITTI odometry sequence {src.parts[-3]}, {src.parent.name} frame {src.stem} "
                      f"(VP2P test sample {int(s['name'])}); map = the LiDAR scan under the eval's seeded random transform",
            "grid_size": GRID_SIZE, "T_c2w": np.linalg.inv(s["T_w2c_gt"]).tolist()}, indent=1))
        print(d, len(s["cloud"]), "points")


if __name__ == "__main__":
    main()
