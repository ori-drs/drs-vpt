# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), data_processing/wai_processing/scripts/covisibility.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.
# Modified for DRS VPT (Oxford Robotics Institute): covisibility from sampled 3-D points (hit counts), for point cloud scenes. See wai_processing.patch.

import logging
import math
import os
import shutil
import traceback
from pathlib import Path

import torch
from argconf import argconf_parse
from tqdm import tqdm
from wai_processing.utils.covis_utils import (
    compute_frustum_intersection,
    load_scene_data,
    project_points_to_views,
    sample_depths_at_reprojections,
)
from wai_processing.utils.state import SceneProcessLock, set_processing_state

from drs_vpt._vendor.mapanything.utils.wai.core import load_data, store_data
from drs_vpt._vendor.mapanything.utils.wai.scene_frame import get_scene_names

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logger = logging.getLogger(__name__)

# -----------------------------
# Sampling / padding helper
# -----------------------------
@torch.no_grad()
def sample_or_pad_points(pts: torch.Tensor, target_n: int = 1500, pad_value: float = 1e6) -> torch.Tensor:
    """
    pts: (N,3)
    returns: (target_n,3)
      - if N >= target_n: uniform random sample without replacement
      - if N <  target_n: pad with a constant far-away point
    """
    assert pts.ndim == 2 and pts.shape[1] == 3
    device = pts.device
    N = pts.shape[0]

    if N >= target_n:
        idx = torch.randperm(N, device=device)[:target_n]
        return pts[idx]

    pad = torch.full((target_n - N, 3), pad_value, device=device, dtype=pts.dtype)
    return torch.cat([pts, pad], dim=0)


# -----------------------------
# Batched counts: fixed i vs many js
# -----------------------------
@torch.no_grad()
def batched_counts_fixed_i(
    pi: torch.Tensor,         # (N,3) (already sampled/padded to TARGET_N)
    pjs: torch.Tensor,        # (K,N,3) (already sampled/padded to TARGET_N)
    r: float = 0.10,
    mode: str = "hits",       # "hits" or "pairs"
) -> torch.Tensor:
    """
    returns: (K,) counts on CPU (int64)
      - mode="pairs": number of (pi,pj) pairs within r
      - mode="hits" : number of pi points that have >=1 pj within r
    """
    assert pi.ndim == 2 and pi.shape[1] == 3
    assert pjs.ndim == 3 and pjs.shape[2] == 3
    assert pjs.shape[1] == pi.shape[0]

    r2 = float(r) * float(r)

    pi2  = (pi * pi).sum(-1).view(1, -1, 1)                 # (1,N,1)
    pjs2 = (pjs * pjs).sum(-1).view(pjs.shape[0], 1, -1)    # (K,1,N)
    dots = torch.einsum("ic,kjc->kij", pi, pjs)             # (K,N,N)

    d2 = (pi2 + pjs2 - 2.0 * dots).clamp_min_(0.0)          # (K,N,N)
    within = d2 <= r2

    if mode == "pairs":
        counts = within.sum(dim=(1, 2))                     # (K,)
    elif mode == "hits":
        counts = within.any(dim=2).sum(dim=1)               # (K,)
    else:
        raise ValueError('mode must be "hits" or "pairs"')

    return counts.to(torch.int64).cpu()


def compute_covisibility(cfg, scene_name: str, overwrite=False):
    scene_root = Path(cfg.root) / scene_name
    scene_meta = load_data(Path(scene_root, "scene_meta.json"), "scene_meta")

    # Delete previous generation
    out_path = scene_root / cfg.out_path
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)

    # Load scene data using the utility function
    scene_data = load_scene_data(cfg, scene_name, device)

    depths = scene_data["depths"]
    depth_h, depth_w = scene_data["depth_dims"]
    valid_depth_masks = scene_data["valid_depth_masks"]
    intrinsics = scene_data["intrinsics"]
    cam2worlds = scene_data["cam2worlds"]
    world_pts3d = scene_data["world_pts3d"]

    num_frames = depths.shape[0]

    # Compute frustum intersection if enabled
    frustum_intersection = compute_frustum_intersection(
        cfg, depths, valid_depth_masks, intrinsics, cam2worlds, device
    )

    # Loop over all the views to compute the pairwise overlap
    pairwise_covisibility = torch.zeros((num_frames, num_frames), device="cpu")
    logger.info("Computing pairwise overlap for each view ...")

    # Process in chunks to avoid OOM
    for idx in tqdm(
        range(num_frames),
        f"Computing exhaustive pairwise covisibility for each view ({scene_name})",
    ):

        if cfg.denominator_mode == "thresh_1000":

            ov_inds = torch.arange(num_frames).to(device)
            if len(ov_inds) == 0:
                continue

            # Process overlapping views in sub-chunks if needed
            overlap_score = torch.zeros((num_frames,), device="cpu")
            ov_chunk_size = 80
            for ov_start in range(0, len(ov_inds), ov_chunk_size):
                ov_end = min(ov_start + ov_chunk_size, len(ov_inds))
                ov_inds_chunk = ov_inds[ov_start:ov_end]
                v_rem = len(ov_inds_chunk)
                if v_rem == 0:
                    continue

                # -----------------------------
                # Full pipeline snippet
                # -----------------------------
                TARGET_N = 1500

                # fixed i (pad negative)
                pi = world_pts3d[idx].to(device).reshape(depth_h, depth_w, -1)[valid_depth_masks[idx]][..., :3]
                pi = sample_or_pad_points(pi, TARGET_N, pad_value=-1e6)

                # contiguous js (pad positive)
                js = list(range(ov_start, ov_end))
                pjs = torch.stack([
                    sample_or_pad_points(
                        world_pts3d[j].to(device).reshape(depth_h, depth_w, -1)[valid_depth_masks[j]][..., :3],
                        TARGET_N,
                        pad_value=1e6,
                    )
                    for j in js
                ], dim=0)  # (K,1500,3)

                # counts in one call
                counts_hits = batched_counts_fixed_i(pi, pjs, r=cfg.get("hit_radius", 1.0), mode="hits")  # (K,)

                overlap_score[ov_inds_chunk.cpu()] = counts_hits.cpu().float()

            # Update the pairwise overlap matrix
            pairwise_covisibility[idx] = overlap_score
            torch.cuda.empty_cache()

        else:
            # Get the remaining views which pass the frustum intersection check
            if cfg.get("perform_frustum_check", True) and frustum_intersection is not None:
                ov_inds = frustum_intersection[idx].argwhere()[:, 0].to(device)
            else:
                ov_inds = torch.arange(num_frames).to(device)
            if len(ov_inds) == 0:
                continue

            # Process overlapping views in sub-chunks if needed
            overlap_score = torch.zeros((num_frames,), device="cpu")
            ov_chunk_size = 40
            for ov_start in range(0, len(ov_inds), ov_chunk_size):
                ov_end = min(ov_start + ov_chunk_size, len(ov_inds))
                ov_inds_chunk = ov_inds[ov_start:ov_end]
                v_rem = len(ov_inds_chunk)
                if v_rem == 0:
                    continue

                # Project the depth map using utility function
                reprojected_pts, valid_mask, _ = project_points_to_views(
                    idx,
                    ov_inds_chunk,
                    depth_h,
                    depth_w,
                    valid_depth_masks,
                    cam2worlds,
                    world_pts3d,
                    intrinsics,
                    device,
                )

                # If any points are valid, compute the covisibility
                if valid_mask.any():
                    # Sample depths at reprojected points
                    depth_lu, expected_depth = sample_depths_at_reprojections(
                        reprojected_pts, depths, ov_inds_chunk, depth_h, depth_w, device
                    )
                    # Calculate reprojection error and depth association threshold
                    reprojection_error = torch.abs(expected_depth - depth_lu)
                    depth_assoc_thres = (
                        cfg.depth_assoc_error_thres
                        + cfg.depth_assoc_rel_error_thres * expected_depth
                        - math.log(0.5) * cfg.depth_assoc_error_temp
                    )
                    valid_depth_projection = (
                        reprojection_error < depth_assoc_thres
                    ) & valid_mask

                    # Calculate covisibility score based on denominator mode
                    if cfg.denominator_mode == "valid_target_depth":
                        # divide by the number of valid depth points in the target view
                        comp_covisibility_score = valid_depth_projection.sum(
                            [1, 2]
                        ) / valid_depth_masks[ov_inds_chunk].sum([1, 2]).clamp(1)
                        comp_covisibility_score = comp_covisibility_score.clamp(
                            0, 1
                        )  # in case in the target view there more valid depth points than in the source view
                    elif cfg.denominator_mode == "full":
                        # divide by the total number of pixels
                        comp_covisibility_score = valid_depth_projection.sum([1, 2]) / (
                            depth_h * depth_w
                        )
                    else:
                        raise NotImplementedError(
                            f"denominator_mode not supported: {cfg.denominator_mode}"
                        )
                    overlap_score[ov_inds_chunk.cpu()] = comp_covisibility_score.cpu()

            # Update the pairwise overlap matrix
            pairwise_covisibility[idx] = overlap_score

            # Free memory
            torch.cuda.empty_cache()

    def symmetrize_and_row_normalize(C: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        C_sym = torch.minimum(C, C.T)
        d = torch.diag(C_sym).clamp_min(eps)
        return C_sym / d[:, None]

    # Hit counts are made symmetric and normalised by each frame's own count; the depth-reprojection scores are
    # already in [0, 1] and stay as computed (asymmetric), as in MapAnything.
    if cfg.denominator_mode == "thresh_1000":
        pairwise_covisibility = symmetrize_and_row_normalize(pairwise_covisibility)

    mmap_store_name = store_data(
        scene_root / cfg.out_path / "pairwise_covisibility.npy",
        pairwise_covisibility,
        "mmap",
    )

    # Update the scene meta
    scene_modalities = scene_meta["scene_modalities"]
    scene_modalities["pairwise_covisibility"] = {
        "scene_key": f"{cfg.out_path}/{mmap_store_name}",
        "format": "mmap",
    }
    store_data(scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    import sys

    logger.debug("Command line arguments:")
    for i, arg in enumerate(sys.argv):
        logger.debug(f"  [{i}]: {arg}")

    cfg = argconf_parse()
    if cfg.get("root") is None:
        raise ValueError(
            "Specify the root via: 'python scripts/covisibility.py root=<root_path>'"
        )
    if cfg.get("frame_modalities") is None:
        raise ValueError("Specify the modality to use for depth in frame_modalities'")

    logger.info("Running covisibility using config:")
    for key, value in dict(cfg).items():
        logger.info(f"  {key}: {value}")

    overwrite = cfg.get("overwrite", False)
    if overwrite:
        logger.warning("Careful: Overwrite enabled!")

    device = cfg.get("device", "cuda")
    scene_names = get_scene_names(
        cfg, shuffle=cfg.get("random_scene_processing_order", True)
    )
    logger.info(f"processing: {len(scene_names)} scenes")
    logger.debug(f"scene_names = {scene_names}")

    for scene_name in tqdm(scene_names, "Processing scenes"):
        try:
            scene_root = Path(cfg.root) / scene_name
            with SceneProcessLock(scene_root):
                logger.info(f"Processing: {scene_name}")
                set_processing_state(scene_root, "covisibility", "running")
                compute_covisibility(cfg, scene_name, overwrite=overwrite)
                set_processing_state(scene_root, "covisibility", "finished")
        except Exception:
            logging.error(f"Computing covisibility failed on scene '{scene_name}'")
            trace_message = traceback.format_exc()
            logger.error(trace_message)
            set_processing_state(
                scene_root, "covisibility", "failed", message=trace_message
            )
            continue
