"""
Core inference invariants.

These tests define the minimum contract for the public-facing API:
  - A user provides a numpy RGB image and an (N, 3) XYZ point cloud.
  - The model returns a valid 4×4 camera pose.
  - The result is deterministic across identical calls.

The test image is the Hub KITTI demo image unless VPT_TEST_IMAGE points to another file.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.smoke, pytest.mark.requires_checkpoint]


def _test_image_path() -> Path:
    override = os.environ.get("VPT_TEST_IMAGE")
    if override:
        p = Path(override)
        if not p.exists():
            pytest.skip(f"VPT_TEST_IMAGE path does not exist: {p}")
        return p
    from drs_vpt.hub import load_sample

    return Path(load_sample("kitti")["frames"][0]["img_path"])


def _small_cloud(seed: int = 42, n: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-5, 5, (n, 3)).astype(np.float32)


_K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)


def test_localize_returns_valid_pose(model):
    """Raw numpy inputs → 4×4 float32 pose, no NaN or Inf."""
    from conftest import localize_pose

    pose = localize_pose(model, _test_image_path(), _K, _small_cloud(), grid_size=0.02)

    assert pose.shape == (4, 4), f"Expected (4, 4), got {pose.shape}"
    assert pose.dtype == np.float32
    assert not np.any(np.isnan(pose)), "Pose contains NaN"
    assert not np.any(np.isinf(pose)), "Pose contains Inf"
    # Rotation block should be approximately a valid rotation matrix
    R = pose[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-3, err_msg="R is not orthogonal")
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-3, err_msg="det(R) != 1")


def test_localize_is_deterministic(model):
    """Identical inputs produce identical poses across two forward passes."""
    from conftest import localize_pose

    image_path = str(_test_image_path())
    cloud = _small_cloud(seed=0)

    def _run():
        return localize_pose(model, image_path, _K, cloud, grid_size=0.02)

    pose_a = _run()
    pose_b = _run()
    np.testing.assert_allclose(pose_a, pose_b, atol=1e-5,
                               err_msg="Model produced different poses on identical inputs")


def test_cloud_output_shape_and_type(model):
    """The voxelized cloud dict has the keys expected by the model."""
    from drs_vpt.inference import _process_cloud

    cloud_dict = _process_cloud(_small_cloud(), grid_size=0.02)
    required_keys = {"coord", "grid_coord", "color", "inverse", "offset", "feat"}
    missing = required_keys - set(cloud_dict.keys())
    assert not missing, f"cloud dict missing keys: {missing}"
    assert isinstance(cloud_dict["coord"], torch.Tensor)
    assert cloud_dict["coord"].shape[-1] == 3, "coord should be (N, 3)"


def test_predicted_intrinsics(model):
    """Without K the model uses intrinsics from its ray map: close to K.npy, and the pose still localizes."""
    from drs_vpt.hub import load_sample
    from drs_vpt.inference import _image_view, encode_map, localize, read_image, resize_to_model
    from drs_vpt.viz import pose_error

    sample = load_sample("7scenes")
    frame = sample["frames"][0]
    image = read_image(frame["img_path"])
    assert "camera_intrinsics" not in _image_view(model, image, None)[0]
    out = localize(model, image, encode_map(model, sample["cloud"], sample["grid_size"], max_points=80_000))

    K_given = resize_to_model(image, frame["K"])[1]
    np.testing.assert_allclose(np.diag(out["K"])[:2], np.diag(K_given)[:2], rtol=0.1)
    np.testing.assert_allclose(out["K"][:2, 2], K_given[:2, 2], atol=10)
    rot_err, trans_err = pose_error(out["T_w2c"], frame["T_c2w"])
    assert rot_err < 5 and trans_err < 0.25, f"GT error {rot_err:.2f} deg, {trans_err:.3f} m"


def test_reusing_map_encoding_changes_nothing(model):
    """One encode_map reused for several localize calls == a fresh encoding per call."""
    from drs_vpt.hub import load_sample
    from drs_vpt.inference import encode_map, localize, read_image

    s = load_sample("iphone")
    f = s["frames"][0]
    image = read_image(f["img_path"])
    fresh = localize(model, image, encode_map(model, s["cloud"], s["grid_size"], max_points=80_000), f["K"])
    map_encoding = encode_map(model, s["cloud"], s["grid_size"], max_points=80_000)
    for _ in range(2):
        out = localize(model, image, map_encoding, f["K"])
        np.testing.assert_allclose(out["T_w2c"], fresh["T_w2c"], atol=1e-6)
        np.testing.assert_allclose(out["T_w2c_coarse"], fresh["T_w2c_coarse"], atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")
def test_mps_matches_cpu(model):
    """With the model on MPS (cloud encoder kept on the CPU), the pose matches the CPU result."""
    from drs_vpt.hub import load_model, load_sample
    from drs_vpt.inference import encode_map, localize, read_image

    s = load_sample("iphone")
    f = s["frames"][0]
    image = read_image(f["img_path"])
    cpu = localize(model, image, encode_map(model, s["cloud"], s["grid_size"], max_points=80_000), f["K"])
    mps_model = load_model(os.environ.get("VPT_CHECKPOINT"), device="mps")
    assert next(mps_model.cloud_encoder.parameters()).device.type == "cpu"
    mps = localize(mps_model, image, encode_map(mps_model, s["cloud"], s["grid_size"], max_points=80_000),
                   f["K"])
    np.testing.assert_allclose(mps["T_w2c"], cpu["T_w2c"], atol=1e-4)
