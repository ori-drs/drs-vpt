#!/usr/bin/env python3
"""Pick representative samples from a finished benchmark run and export them as regression fixtures.

    python scripts/eval.py --dataset kitti --data /data/kitti_vp2p_eval_data --out eval_output/kitti
    python scripts/export_regression_set.py --dataset kitti --data /data/kitti_vp2p_eval_data \\
        --results eval_output/kitti --out /data/drsvpt_regression

Samples are grouped (KITTI: sequence and camera; nuScenes: --bins contiguous index ranges, as the names carry no
scene), and --n samples are taken round-robin over the groups at evenly spaced quantiles of difficulty,
max(RRE / 5 deg, RTE / 2 m). Only samples localized within 5 deg and 2 m are picked: a diverged pose has no reason
to land in the same place on other hardware, so it cannot serve as a fixture.

Each sample is written to <out>/<dataset>/<name>/ in the load_sample layout: image.<ext>, K.npy, cloud.npy (the
perturbed cloud exactly as the model sees it) and meta.json (grid_size, T_c2w, the group, why it was picked and its
benchmark errors). The fixtures need no dataset, loader or seeding; record their expected outputs with
tests/record_regression.py.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from drs_vpt.eval import GRID_SIZE, RRE_THRESHOLD_DEG, RTE_THRESHOLD_M, SAMPLES


def difficulty(r: dict) -> float:
    return max(r["rre_deg"] / RRE_THRESHOLD_DEG, r["rte_m"] / RTE_THRESHOLD_M)


def group_of(dataset: str, r: dict, index: int, n: int, bins: int) -> str:
    if dataset == "kitti":
        parts = Path(r["image_path"]).parts  # .../sequences/<seq>/<image_2|image_3>/<frame>.png
        return f"{parts[-3]}/{parts[-2]}"
    return f"bin{index * bins // n}"


def select(dataset: str, results: list, bins: int, n: int) -> dict:
    """{name: (group, reason)} for n samples spread over the groups and over the error range, all localized within
    the recall thresholds: round-robin over the groups, taking evenly spaced quantiles of difficulty in each."""
    results = sorted(results, key=lambda r: r["name"])
    groups = {}
    for i, r in enumerate(results):
        if difficulty(r) < 1:
            groups.setdefault(group_of(dataset, r, i, len(results), bins), []).append(r)
    groups = {g: sorted(rs, key=difficulty) for g, rs in sorted(groups.items())}
    per_group = -(-n // len(groups))
    picked = {}
    for k in range(per_group):
        q = (k + 0.5) / per_group
        for g, rs in groups.items():
            if len(picked) < n:
                picked.setdefault(rs[int(q * (len(rs) - 1))]["name"], (g, f"q{round(100 * q)}"))
    return picked


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(SAMPLES))
    p.add_argument("--data", required=True, help="Dataset root, as for scripts/eval.py")
    p.add_argument("--results", required=True, help="Folder of per-sample JSONs written by scripts/eval.py")
    p.add_argument("--out", required=True, help="Fixture root; samples go to <out>/<dataset>/<name>/")
    p.add_argument("--n", type=int, default=10, help="Number of samples (default: 10)")
    p.add_argument("--bins", type=int, default=5, help="nuScenes: number of index ranges to sample from")
    args = p.parse_args()

    results = [json.loads(f.read_text()) for f in Path(args.results).glob("*.json") if f.name != "summary.json"]
    by_name = {r["name"]: r for r in results}
    picked = select(args.dataset, results, args.bins, args.n)
    print(f"{len(picked)} of {len(results)} {args.dataset} samples")

    for s in SAMPLES[args.dataset](args.data, names=picked):
        r, (group, reason) = by_name[s["name"]], picked[s["name"]]
        d = Path(args.out) / args.dataset / s["name"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(s["image_path"], d / f"image{Path(s['image_path']).suffix}")
        np.save(d / "K.npy", np.asarray(s["K"], dtype=np.float64))
        np.save(d / "cloud.npy", np.asarray(s["cloud"], dtype=np.float32))
        meta = {"dataset": args.dataset, "source_image": r["image_path"], "group": group, "reason": reason,
                "grid_size": GRID_SIZE, "T_c2w": np.linalg.inv(np.asarray(s["T_w2c_gt"], dtype=np.float64)).tolist(),
                "benchmark": {"rre_deg": r["rre_deg"], "rte_m": r["rte_m"]}}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  {s['name']}  {group:12s} {reason:6s}  RRE {r['rre_deg']:.3f} deg  RTE {r['rte_m']:.3f} m")


if __name__ == "__main__":
    main()
