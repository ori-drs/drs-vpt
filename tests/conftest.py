"""
Shared fixtures and helpers for DRS VPT integration tests.

The model (fulkast/drs_vpt), the KITTI / 7Scenes samples and the benchmark regression set
(fulkast/drs_vpt_demo_data) are downloaded from the Hugging Face Hub on first use. Optional environment variables:

    VPT_CHECKPOINT      local checkpoint (save_pretrained dir or training .pth) instead of the Hub model
    VPT_DEVICE          device for the model (default cpu); fixtures recorded on one device may drift on another
    VPT_REGRESSION_DATA local benchmark fixture set written by scripts/export_regression_set.py
                        (<root>/<dataset>/<name>/) instead of the Hub one
"""

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Marker declarations
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: fast sanity checks, no dataset required")
    config.addinivalue_line("markers", "benchmark: full-dataset eval, slow — reproduces paper numbers")
    config.addinivalue_line("markers", "requires_checkpoint: needs the model (Hub download or VPT_CHECKPOINT)")
    config.addinivalue_line("markers", "requires_data: needs a data sample (Hub download or a dataset env var)")


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    from drs_vpt.hub import load_model

    return load_model(os.environ.get("VPT_CHECKPOINT"), device=os.environ.get("VPT_DEVICE", "cpu"))


# ---------------------------------------------------------------------------
# Inference helpers (thin wrappers around drs_vpt.inference)
# ---------------------------------------------------------------------------

def localize_pose(model: torch.nn.Module, image_path, K, cloud_xyz: np.ndarray, grid_size: float) -> np.ndarray:
    """(4, 4) float32 world-to-camera pose at the finest coarse-to-fine level, for one image + cloud."""
    from drs_vpt.inference import encode_map, localize, read_image

    return localize(model, read_image(image_path), encode_map(model, cloud_xyz, grid_size), K)["T_w2c"]


def hub_sample_pose(model: torch.nn.Module, name: str) -> np.ndarray:
    """localize_pose for the first frame of a Hub demo sample (downloaded and cached on first use)."""
    from drs_vpt.hub import load_sample

    s = load_sample(name)
    f = s["frames"][0]
    return localize_pose(model, f["img_path"], f["K"], s["cloud"], s["grid_size"])


# ---------------------------------------------------------------------------
# Benchmark regression fixtures (scripts/export_regression_set.py)
# ---------------------------------------------------------------------------

EXPECTED = Path(__file__).parent / "assets" / "regression"


def regression_fixture_dirs() -> list:
    """Benchmark sample folders (<root>/<dataset>/<name>/), from VPT_REGRESSION_DATA if set, else the Hub regression
    set; empty when neither is available."""
    root = os.environ.get("VPT_REGRESSION_DATA")
    if root is None:
        from drs_vpt.hub import regression_data_dir
        try:
            root = regression_data_dir()
        except Exception as e:  # offline, or no access to the Hub repo
            print(f"[conftest] benchmark regression set unavailable: {e}")
            return []
    if not Path(root).is_dir():
        return []
    return sorted(d for d in Path(root).glob("*/*") if (d / "meta.json").exists())


def regression_fixture_result(model: torch.nn.Module, d: Path) -> dict:
    """Localize one fixture in fp32 (so results do not depend on the GPU's autocast choice) and score it."""
    from drs_vpt.hub import load_sample
    from drs_vpt.inference import encode_map, localize, read_image
    from drs_vpt.viz import pose_error

    s = load_sample(str(d))
    f = s["frames"][0]
    T_w2c = localize(model, read_image(f["img_path"]), encode_map(model, s["cloud"], s["grid_size"]), f["K"],
                     autocast=contextlib.nullcontext())["T_w2c"]
    rre, rte = pose_error(T_w2c, f["T_c2w"])
    return {"T_w2c": np.asarray(T_w2c).tolist(), "rre_deg": rre, "rte_m": rte}
