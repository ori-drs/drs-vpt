# Copyright (C) 2024-present Oxford Robotics Institute.
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Benchmark evaluation: single-image localization on the KITTI and nuScenes image-to-point-cloud test sets.

    model = load_model(device="cuda")
    results = evaluate(model, kitti_samples("/data/kitti_vp2p_eval_data"), out_dir="eval_output/kitti")
    print(summarize(results))

Each sample is a camera image and a LiDAR cloud in the camera frame, moved by a random rigid transform: up to 10 m in
x and z and any rotation about the vertical (camera y) axis, drawn with the sample index as the seed so every run
sees the same perturbations. The model localizes the image in the cloud from scratch, with no pose prior. Errors are on the camera pose in the cloud frame:
    RRE  rotation error in degrees, the angle between the predicted and true camera rotations
    RTE  translation error in metres, the distance between the predicted and true camera centres
Recall counts samples with RRE < 5 deg and RTE < 2 m.
"""

import json
import os
import random
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
from drs_vpt.inference import localize_prepared, pick_autocast, prepare_input, read_image
from drs_vpt.viz import pose_error

RRE_THRESHOLD_DEG = 5.0
RTE_THRESHOLD_M = 2.0
GRID_SIZE = 0.02  # voxel size of both LiDAR benchmarks, in metres
MAX_TRANSLATION_M = 10.0  # in x and z of the camera frame; the CorrI2P / VP2P protocol


def random_perturbation(seed: int) -> np.ndarray:
    """The 4x4 rigid transform applied to a camera-frame cloud: a translation up to MAX_TRANSLATION_M in x and z, then
    any rotation about y: the distribution the KITTI loader draws from (kitti_vp2p.generate_random_transform)."""
    rng = random.Random(seed)
    t = [rng.uniform(-MAX_TRANSLATION_M, MAX_TRANSLATION_M), 0.0, rng.uniform(-MAX_TRANSLATION_M, MAX_TRANSLATION_M)]
    yaw = rng.uniform(-2.0 * np.pi, 2.0 * np.pi)
    P = np.identity(4)
    P[0:3, 0:3] = [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    P[0:3, 3] = t
    return P


class KittiSet:
    """KITTI odometry test samples (sequences 9 and 10) from the VP2P evaluation data, loaded by name.

    The random transform of each sample is drawn with the sample index as the seed, so every run sees the same
    perturbations. The loader is built on first use, so the set can be sent to data loader workers.
    """

    def __init__(self, root: str):
        self.root = root
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            from drs_vpt.datasets.kitti_vp2p import kitti_pc_img_dataset
            self._dataset = kitti_pc_img_dataset(self.root, "val", 1_000_000)
        return self._dataset

    def names(self) -> list:
        return [f"{i:06d}" for i in range(len(self.dataset))]

    def load(self, name: str) -> dict:
        i = int(name)
        random.seed(i)
        np.random.seed(i)
        data = self.dataset[i]
        # The cloud is P_random @ x_cam and the dataset returns P = inv(P_random), so P maps cloud to camera.
        return {"name": f"{i:06d}", "image_path": data["img_path"], "K": data["K"].numpy(),
                "cloud": data["pc"].T.numpy()[:, :3].astype(np.float32), "T_w2c_gt": data["P"].numpy()}


class NuscenesSet:
    """nuScenes test samples from the set data_processing/nuscenes_preprocess.py builds: img/<name>.png, K/<name>.npy
    and PC/<name>.npy (3 or more x N, camera frame), loaded by name. The cloud is moved by
    random_perturbation(int(name)); the stored P/<name>.npy (the LiDAR-to-camera transform) is not used.
    """

    def __init__(self, root: str):
        self.root = Path(root)

    def names(self) -> list:
        return sorted({x.split(".")[0] for x in os.listdir(self.root / "img") if "clou" not in x})

    def load(self, name: str) -> dict:
        from drs_vpt.datasets.utils import apply_transform_from_left

        P = random_perturbation(int(name))
        raw = np.load(self.root / "PC" / f"{name}.npy").T.astype(np.float32)[:, :3]
        # The cloud is P @ x_cam, so the camera-to-cloud pose is P.
        return {"name": name, "image_path": str(self.root / "img" / f"{name}.png"),
                "K": np.load(self.root / "K" / f"{name}.npy"),
                "cloud": apply_transform_from_left(P, raw).astype(np.float32), "T_w2c_gt": np.linalg.inv(P)}


SETS = {"kitti": KittiSet, "nuscenes": NuscenesSet}


def _select(names: list, limit: Optional[int], subset: Optional[Iterable[str]]) -> list:
    return sorted(subset) if subset is not None else names[:limit]


def kitti_samples(root: str, limit: Optional[int] = None, names: Optional[Iterable[str]] = None) -> Iterator[dict]:
    """KITTI test samples (see KittiSet). names: only these samples (as yielded in "name"). Yields {"name",
    "image_path", "K", "cloud", "T_w2c_gt"}."""
    data = KittiSet(root)
    for name in _select(data.names(), limit, names):
        yield data.load(name)


def nuscenes_samples(root: str, limit: Optional[int] = None, names: Optional[Iterable[str]] = None) -> Iterator[dict]:
    """nuScenes test samples (see NuscenesSet). names: only these samples. Yields {"name", "image_path", "K",
    "cloud", "T_w2c_gt"}."""
    data = NuscenesSet(root)
    for name in _select(data.names(), limit, names):
        yield data.load(name)


SAMPLES = {"kitti": kitti_samples, "nuscenes": nuscenes_samples}


def _prepare(sample: dict) -> dict:
    """A sample with its model input (CPU work: image resize, cloud voxelization) and without the raw cloud."""
    return {"name": sample["name"], "image_path": str(sample["image_path"]),
            "T_w2c_gt": np.asarray(sample["T_w2c_gt"]),
            "input": prepare_input(read_image(sample["image_path"]), sample["cloud"], sample["K"], GRID_SIZE)}


class _PreparedSet:
    """Map-style dataset of prepared samples, for a torch DataLoader."""

    def __init__(self, data, names: list):
        self.data, self.names = data, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        return _prepare(self.data.load(self.names[i]))


def _as_list(batch):
    return batch


def _batched(items: Iterable, n: int) -> Iterator[list]:
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _run(model, batches: Iterable[list], out: Optional[Path], results: list, verbose: bool) -> list:
    """Localize batches of prepared samples, appending one result per sample (and writing it under out)."""
    autocast, _ = pick_autocast(model.scale_token.device)
    for batch in batches:
        preds = localize_prepared(model, [x["input"] for x in batch], autocast=autocast)
        for x, pred in zip(batch, preds):
            rre, rte = pose_error(pred["T_w2c"], np.linalg.inv(np.asarray(x["T_w2c_gt"], dtype=np.float64)))
            result = {"name": x["name"], "image_path": x["image_path"], "rre_deg": rre, "rte_m": rte,
                      "T_w2c_pred": pred["T_w2c"].tolist(), "T_w2c_gt": np.asarray(x["T_w2c_gt"]).tolist()}
            if out:
                (out / f"{x['name']}.json").write_text(json.dumps(result, indent=2))
            results.append(result)
        if verbose:
            s = summarize(results)
            print(f"{batch[-1]['name']}: n {s['n']}  recall {s['recall']:.4f}  "
                  f"median {s['rre_median_deg']:.3f} deg / {s['rte_median_m']:.3f} m", flush=True)
    return results


def _resume(out_dir: Optional[str], names: Iterable[str]) -> tuple:
    """(out, the results already written for these names, the names still to run)."""
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    done, todo = [], []
    for name in names:
        path = out / f"{name}.json" if out else None
        if path and path.exists():
            done.append(json.loads(path.read_text()))
        else:
            todo.append(name)
    return out, done, todo


def evaluate(model, samples, out_dir: Optional[str] = None, verbose: bool = True, batch_size: int = 1) -> list:
    """Localize every sample and return one result dict per sample.

    out_dir: if given, each result is written to <out_dir>/<name>.json, and samples that already have one are read
    back instead of re-run, so an interrupted evaluation resumes where it stopped.
    batch_size: samples per forward pass. A sample's result does not depend on the batch size, up to floating-point
    differences of the batched GPU kernels. For a whole benchmark, evaluate_benchmark also overlaps the CPU work.
    """
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    results = []

    def todo():
        for sample in samples:
            path = out / f"{sample['name']}.json" if out else None
            if path and path.exists():
                results.append(json.loads(path.read_text()))
            else:
                yield _prepare(sample)

    return _run(model, _batched(todo(), batch_size), out, results, verbose)


def evaluate_benchmark(model, dataset: str, root: str, limit: Optional[int] = None, out_dir: Optional[str] = None,
                       batch_size: int = 8, num_workers: int = 8, verbose: bool = True) -> list:
    """evaluate on a whole benchmark set ("kitti" or "nuscenes"), with num_workers processes loading and voxelizing
    the next batches while the GPU runs. Returns the results of the first limit samples (all with None)."""
    import torch

    data = SETS[dataset](root)
    out, done, todo = _resume(out_dir, data.names()[:limit])
    if verbose and done:
        print(f"{len(done)} samples already in {out}, {len(todo)} to run", flush=True)
    loader = torch.utils.data.DataLoader(_PreparedSet(data, todo), batch_size=batch_size, num_workers=num_workers,
                                         collate_fn=_as_list, persistent_workers=False,
                                         prefetch_factor=2 if num_workers > 0 else None)
    return _run(model, loader, out, done, verbose)


def summarize(results: list) -> dict:
    """Recall at RRE < 5 deg and RTE < 2 m, and the mean and median errors."""
    rre = np.array([r["rre_deg"] for r in results])
    rte = np.array([r["rte_m"] for r in results])
    return {
        "n": len(results),
        "recall": float(np.mean((rre < RRE_THRESHOLD_DEG) & (rte < RTE_THRESHOLD_M))),
        "rre_mean_deg": float(rre.mean()), "rre_median_deg": float(np.median(rre)),
        "rte_mean_m": float(rte.mean()), "rte_median_m": float(np.median(rte)),
    }
