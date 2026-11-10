"""DRS VPT inference: localize images in a point cloud map, and calibrate a camera to a LiDAR.

    model = load_model(device="cuda")
    map_encoding = encode_map(model, cloud_xyz, grid_size=0.02)   # once per map
    out = localize(model, image, map_encoding, K)                 # K=None: intrinsics from the predicted ray map
    out["T_w2c"]                                                  # x_cam = R @ x_cloud + t

The cloud is an (N, 3) float32 array in metres; the image is an (H, W, 3) uint8 RGB array (read_image loads a file);
K is the 3x3 pinhole matrix of the image at its original resolution. Camera-LiDAR calibration:
calibrate(model, [(image, K, scan), ...]) -> T_cam_lidar.
"""

import time

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose
from scipy.spatial.transform import Rotation

# Cloud dict keys that must stay integer tensors
_INT_KEYS = {
    "offset", "inverse", "batch", "grid_coord",
    "serialized_order", "serialized_inverse", "pooling_inverse",
}
_IGNORE_KEYS = {"instance", "idx", "true_shape", "data_norm_type"}


def _cloud_features(points_xyz: np.ndarray, grid_size: float = 0.02, max_points: int = 0) -> dict:
    """Voxelize an (N, 3) cloud into one sample's Sonata input (CPU only, not yet collated).

    grid_size is the voxel size in metres (outdoor LiDAR ~0.02, indoor scans 0.005-0.01).
    max_points > 0 randomly subsamples the cloud first (deterministically) to bound memory.
    """
    from drs_vpt._vendor import sonata
    from drs_vpt.datasets.utils import extract_view

    if max_points > 0 and len(points_xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(points_xyz), max_points, replace=False)
        points_xyz = points_xyz[idx]
    # Sonata's GridSample (mode="train") keeps a random point per voxel; seed for repeatable output.
    np.random.seed(0)
    cloud_view = extract_view(np.eye(4, dtype=np.float32), points_xyz, np.array([grid_size], dtype=np.float32))
    lidar_keys = {"coord", "normal", "color", "grid_size", "scene_scale"}
    return sonata.transform.default()({k: cloud_view[k] for k in lidar_keys})


def _process_cloud(points_xyz: np.ndarray, grid_size: float = 0.02, max_points: int = 0) -> dict:
    """Voxelize an (N, 3) cloud into the collated Sonata input dict (a batch of one)."""
    from drs_vpt._vendor import sonata

    return sonata.data.collate_fn([_cloud_features(points_xyz, grid_size, max_points)])


def read_image(path) -> np.ndarray:
    """An image file as an (H, W, 3) uint8 RGB array, turned upright by its EXIF orientation."""
    return np.array(exif_transpose(Image.open(path)).convert("RGB"))


def resize_to_model(image: np.ndarray, K=None) -> tuple:
    """The image cropped and resized to the model's input resolution (the 518 px resolution closest to its aspect
    ratio), and K adjusted to match (None when K is None)."""
    from drs_vpt._vendor.mapanything.utils.cropping import crop_resize_if_necessary
    from drs_vpt._vendor.mapanything.utils.image import find_closest_aspect_ratio

    img = Image.fromarray(image)
    resized = crop_resize_if_necessary(img, resolution=find_closest_aspect_ratio(img.width / img.height, 518),
                                       intrinsics=K)
    return np.asarray(resized[0]), (resized[1] if K is not None else None)


def _image_input(image: np.ndarray, K) -> tuple:
    """The normalized (3, H, W) image tensor at the model's resolution, and K at that resolution (None if K is
    None). CPU only."""
    import torchvision.transforms as tvf
    from drs_vpt._vendor.uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    resized, new_K = resize_to_model(image, K)
    norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    to_tensor = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=norm.mean, std=norm.std)])
    return to_tensor(Image.fromarray(resized)).float(), new_K


def _image_view(model, image: np.ndarray, K) -> list:
    """The model's input view for an image, with K at the model's resolution (no K: none is given to the model)."""
    img, new_K = _image_input(image, K)
    return _batch_view(model, [img], [new_K])


def _batch_view(model, images: list, Ks: list) -> list:
    """The model's input view for a batch of (3, H, W) image tensors of one shape, with their K at the model's
    resolution (all None: no K is given to the model)."""
    device = model.scale_token.device
    img = torch.stack(images).to(device)
    view = {"img": img, "true_shape": np.int32([img.shape[-2:]] * len(images)), "idx": 0, "instance": "0",
            "data_norm_type": ["dinov2"]}
    if Ks[0] is not None:
        view["camera_intrinsics"] = torch.as_tensor(np.stack(Ks), dtype=torch.float32).to(device)
    return [view]


def _encode_cloud_batch(model, cloud_features: list) -> dict:
    """Collate the per-sample Sonata inputs (from _cloud_features) and encode them as one batch of maps."""
    from drs_vpt._vendor import sonata

    cloud = sonata.data.collate_fn(list(cloud_features))
    device = next(model.cloud_encoder.parameters()).device
    tensors = {k: (v.long().to(device) if k in _INT_KEYS else v.float().to(device))
               for k, v in cloud.items() if isinstance(v, torch.Tensor)}
    with torch.inference_mode():
        return model.encode_map(tensors)


def encode_map(model, cloud_xyz: np.ndarray, grid_size: float = 0.02, max_points: int = 0) -> dict:
    """Voxelize and encode a map cloud once; pass the result to localize for any number of images.

    The cloud encoder runs where load_model put it (the CPU when the model is on MPS).
    """
    return _encode_cloud_batch(model, [_cloud_features(cloud_xyz, grid_size=grid_size, max_points=max_points)])


def pick_autocast(device) -> tuple:
    """Return (autocast context, description): fp16 only on GPUs where it is faster than fp32.

    GPUs without tensor cores (e.g. Quadro T2000) run fp16 ~4x slower, so a quick matmul benchmark
    decides. CPU always runs fp32 (spconv has no bf16 kernels).
    """
    if torch.device(device).type != "cuda":
        return torch.autocast("cpu", enabled=False), f"fp32 ({torch.device(device).type})"

    def bench(dtype):
        a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
        a @ a
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            a @ a
        torch.cuda.synchronize()
        return time.perf_counter() - t

    fp16_fast = bench(torch.float16) < bench(torch.float32)
    torch.cuda.empty_cache()
    return (torch.autocast("cuda", dtype=torch.float16, enabled=fp16_fast),
            "fp16 autocast" if fp16_fast else "fp32 (fp16 is slow on this GPU)")


def _pose(quat_xyzw: torch.Tensor, trans: torch.Tensor) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_quat(quat_xyzw.float().cpu().numpy()).as_matrix()
    T[:3, 3] = trans.float().cpu().numpy()
    return T


def localize(model, image: np.ndarray, map_encoding: dict, K=None, autocast=None) -> dict:
    """Localize one image in an encoded map: world-to-camera poses (4x4 float32) of the camera.

    image: (H, W, 3) uint8 RGB array (read_image loads a file).
    K: 3x3 intrinsics of the image at its original resolution; None uses the K recovered from the predicted ray map.

    T_w2c        final pose: finest coarse-to-fine level after the learned optimizer
    T_w2c_coarse direct transformer prediction, before optimization
    K            3x3 intrinsics the aligner used, at the model's resolution (given, or predicted from the ray map)
    image_shape  (H, W) of the image at the model's resolution (the resolution K refers to)
    raw          the model's full output, for anything else you want to inspect
    """
    if map_encoding is None:
        raise ValueError("localize needs a map: pass encode_map(model, cloud, grid_size)")
    image_views = _image_view(model, image, K)
    if autocast is None:
        autocast, _ = pick_autocast(image_views[0]["img"].device)
    with torch.inference_mode(), autocast:
        raw = model(image_views, align_in_predicted_scale=False, map_encoding=map_encoding)
    return {**_batch_poses(raw, image_views)[0], "raw": raw}


def _batch_poses(raw: dict, image_views: list) -> list:
    """Per-sample poses from a forward pass: T_w2c, T_w2c_coarse, K and image_shape (see localize)."""
    cloud = raw["cloud"]
    K = raw["image_views"][0]["camera_intrinsics_used"].float().cpu().numpy()
    return [{
        # [batch b][c2f level 2] is the finest level
        "T_w2c": _pose(cloud["cloud_quats_opt"][b][2], cloud["cloud_trans_opt"][b][2]),
        "T_w2c_coarse": _pose(cloud["cloud_quats"][b], cloud["cloud_trans"][b]),
        "K": K[b],
        "image_shape": tuple(image_views[0]["img"].shape[-2:]),
    } for b in range(len(K))]


def prepare_input(image: np.ndarray, cloud_xyz: np.ndarray, K=None, grid_size: float = 0.02,
                  max_points: int = 0) -> dict:
    """The CPU part of localizing an image in its own cloud: the resized, normalized image and the voxelized cloud.
    Picklable, so data loader workers can prepare inputs for localize_prepared while the GPU runs."""
    img, new_K = _image_input(image, K)
    return {"img": img, "K": new_K, "cloud": _cloud_features(cloud_xyz, grid_size, max_points)}


def localize_prepared(model, inputs: list, autocast=None) -> list:
    """Localize a batch of images, each in its own cloud, from prepare_input outputs: one forward pass per group of
    inputs with the same image shape and with or without K. Returns one dict per input, as localize (without "raw").

    Each sample's result does not depend on the others in the batch, up to floating-point differences of the batched
    GPU kernels.
    """
    if not inputs:
        return []
    if autocast is None:
        autocast, _ = pick_autocast(model.scale_token.device)
    groups = {}
    for i, x in enumerate(inputs):
        groups.setdefault((tuple(x["img"].shape), x["K"] is None), []).append(i)
    results = [None] * len(inputs)
    for indices in groups.values():
        image_views = _batch_view(model, [inputs[i]["img"] for i in indices], [inputs[i]["K"] for i in indices])
        map_encoding = _encode_cloud_batch(model, [inputs[i]["cloud"] for i in indices])
        with torch.inference_mode(), autocast:
            raw = model(image_views, align_in_predicted_scale=False, map_encoding=map_encoding)
        for i, pose in zip(indices, _batch_poses(raw, image_views)):
            results[i] = pose
    return results


def localize_batch(model, images: list, clouds: list, Ks: list = None, grid_size: float = 0.02,
                   max_points: int = 0, autocast=None) -> list:
    """Localize several images, each in its own (N, 3) cloud, in batched forward passes. Ks: one K per image, or
    None to use the predicted intrinsics for all. Returns one dict per image, as localize (without "raw")."""
    Ks = Ks if Ks is not None else [None] * len(images)
    return localize_prepared(model, [prepare_input(im, c, K, grid_size, max_points)
                                     for im, c, K in zip(images, clouds, Ks)], autocast=autocast)


def calibration_record(model, image: np.ndarray, K, scan: np.ndarray, grid_size: float = 0.02,
                       max_points: int = 0, autocast=None) -> dict:
    """One frame for calibrate_records: the normal forward pass of an image against its own scan (an (N, 3) cloud
    in the fixed LiDAR frame, not centred), keeping the aligner's inputs at every coarse-to-fine level (map points in
    metres in the LiDAR frame, their features and confidences, the image features) and the frame's own pose."""
    aligner = model.aligner
    original = aligner.run_optimization
    levels = []

    def spy(**kw):
        levels.append({k: kw[k] for k in ("p3D", "F_ref", "F_query", "camera", "W_ref_query")})
        return original(**kw)

    aligner.run_optimization = spy
    try:
        out = localize(model, image, encode_map(model, scan, grid_size, max_points), K, autocast=autocast)
    finally:
        del aligner.run_optimization
    for r in levels:
        if aligner.normalize_features:
            r["F_ref"] = torch.nn.functional.normalize(r["F_ref"], dim=-1)
    return {"levels": levels, "T_w2c": out["T_w2c"]}


def calibrate_records(model, records: list, iters: int = None) -> dict:
    """Camera-LiDAR calibration: one T_cam_lidar shared by every frame, from the model's own feature alignment.

    records: from calibration_record, one per frame. The aligner is rerun once over all frames: at every level and
    iteration, each frame's feature residuals and Jacobians are evaluated at the same pose, the per-frame normal
    equations (each normalized by its weight sum, as in the single-frame aligner) are summed, and one LM step updates
    the shared pose. It runs from every frame's own estimate as a start and keeps the solution with the lowest final
    joint cost at the finest level.

    T_cam_lidar  4x4 shared calibration (x_cam = R @ x_lidar + t)
    T_per_frame  each frame's own estimate (as localize)
    start        index of the frame estimate the kept solution started from
    start_costs  final finest-level joint cost of the solution from each start
    cost         joint cost after every iteration, per level, of the kept solution
    """
    from drs_vpt._vendor.pixloc.wrappers import Pose
    from drs_vpt._vendor.pixloc.optimization import optimizer_step

    aligner = model.aligner
    systems = [r["levels"] for r in records]
    T_frames = [r["T_w2c"] for r in records]

    def frame_system(T, r):
        res, valid, w_unc, _, J = aligner.cost_fn.residual_jacobian(
            T, r["camera"], r["p3D"], r["F_ref"], r["F_query"], r["W_ref_query"])
        cost, w_loss, _ = aligner.loss_fn((res ** 2).sum(-1))
        weights = w_loss * valid.float()
        if w_unc is not None:
            weights = weights * w_unc
        g, H = aligner.build_system(J, res, weights)
        return g, H, (cost * weights).sum() / weights.sum().clamp_min(1.0)

    def joint_cost(T, level):
        return sum(frame_system(T, s[level])[2] for s in systems).item()

    p3D = systems[0][0]["p3D"]

    def to_pose(T):
        T = torch.as_tensor(T, dtype=p3D.dtype, device=p3D.device)
        return Pose.from_Rt(T[None, :3, :3], T[None, :3, 3])

    def solve(T):
        history = []
        for level in range(len(systems[0])):
            history.append([])
            for _ in range(iters or aligner.num_iterations):
                g = H = 0
                for s in systems:
                    g_f, H_f, _ = frame_system(T, s[level])
                    g, H = g + g_f, H + H_f
                dt, dw = optimizer_step(g, H, 0.01).split([3, 3], dim=-1)
                T = Pose.from_aa(dw, dt) @ T
                history[-1].append(joint_cost(T, level))
        return T, history

    with torch.inference_mode():
        # Run the joint alignment from every frame's own estimate and keep the lowest final (finest-level) cost:
        # the coarse cost barely separates starts, and a single flipped frame estimate must not be the start.
        runs = [solve(to_pose(T)) for T in T_frames]
    final = [h[-1][-1] for _, h in runs]
    start = int(np.argmin(final))
    T, history = runs[start]
    T_cam_lidar = np.eye(4, dtype=np.float32)
    T_cam_lidar[:3, :3] = T.R[0].float().cpu().numpy()
    T_cam_lidar[:3, 3] = T.t[0].float().cpu().numpy()
    return {"T_cam_lidar": T_cam_lidar, "T_per_frame": T_frames, "start": start, "start_costs": final,
            "cost": history}


def calibrate(model, frames: list, grid_size: float = 0.02, max_points: int = 0, iters: int = None,
              autocast=None) -> dict:
    """Camera-LiDAR calibration from frames [(image, K, scan_xyz)], each scan an (N, 3) cloud in the fixed
    LiDAR frame, synced with its image. See calibration_record and calibrate_records."""
    records = [calibration_record(model, image, K, scan, grid_size, max_points, autocast)
               for image, K, scan in frames]
    return calibrate_records(model, records, iters)
