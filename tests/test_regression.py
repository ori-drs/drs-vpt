"""
Regression tests — Hub demo samples plus a representative set from the benchmark runs.

These catch API breakage and accuracy drift without running full eval sets.  Each test:
  1. Loads a single known sample.
  2. Runs the model.
  3. Compares the output against a pre-recorded expected output stored in
     tests/assets/regression/.

The expected poses are generated once by running:
    python tests/record_regression.py

and the resulting .npy files are committed to the repo.  If an expected pose
file does not exist the test is skipped with a clear message.

The KITTI and 7Scenes inputs are the Hub demo samples (fulkast/drs_vpt_demo_data), which store
exactly the image, intrinsics and cloud the model sees. The benchmark set holds ~10 KITTI and ~10
nuScenes samples picked by scripts/export_regression_set.py across sequences / index ranges and across
the error distribution, all localized within the recall thresholds. For each, the pose must stay within
5 mm and 0.05 deg of the recorded one (test_benchmark_pose: for refactors that should change no numbers),
and its errors within a margin of the recorded ones (test_benchmark_accuracy: for intended changes). Optional env vars:
    VPT_CHECKPOINT      local checkpoint instead of the Hub model
    VPT_DEVICE          device for the model (default cpu); record and test on the same device
    VPT_REGRESSION_DATA local benchmark fixture set instead of the Hub one (regression/ in fulkast/drs_vpt_demo_data);
                        the benchmark tests are skipped when neither is available
"""

import json


from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_checkpoint, pytest.mark.requires_data]

ASSETS = Path(__file__).parent / "assets" / "regression"

from conftest import EXPECTED, regression_fixture_dirs, regression_fixture_result

# KITTI and 7Scenes use the Hub demo samples (KITTI VP2P val sample 539; 7Scenes stairs/seq-01/frame-000000).

# How far a pose may move from the recorded one: camera centre and rotation. Different hardware, thread counts and
# library builds sum in a different order, which moves a well-localized pose by far less than this.
POSE_TOL_M = 0.005
POSE_TOL_DEG = 0.05

# How much worse than recorded a benchmark sample may get before test_benchmark_accuracy fails.
RRE_MARGIN_DEG = 0.25
RTE_MARGIN_M = 0.05


def assert_pose_close(got, expected, what: str):
    """The camera centres of two world-to-camera poses within POSE_TOL_M, and their rotations within POSE_TOL_DEG."""
    from drs_vpt.viz import pose_error

    drift_deg, drift_m = pose_error(np.asarray(got, dtype=np.float64),
                                    np.linalg.inv(np.asarray(expected, dtype=np.float64)))
    assert drift_m <= POSE_TOL_M and drift_deg <= POSE_TOL_DEG, \
        f"{what}: pose moved {drift_m * 1000:.1f} mm / {drift_deg:.3f} deg from the recorded one"


def _expected(name: str) -> np.ndarray:
    path = ASSETS / f"{name}_expected_pose.npy"
    if not path.exists():
        pytest.skip(
            f"No recorded pose for '{name}'.  "
            f"Run  python tests/record_regression.py  to generate it."
        )
    return np.load(path)


# ---------------------------------------------------------------------------
# KITTI
# ---------------------------------------------------------------------------

def test_kitti_regression(model):
    from conftest import hub_sample_pose

    pose = hub_sample_pose(model, "kitti")

    expected = _expected("kitti")
    assert_pose_close(pose, expected, "KITTI demo sample")


# ---------------------------------------------------------------------------
# 7Scenes
# ---------------------------------------------------------------------------

def test_7scenes_regression(model):
    from conftest import hub_sample_pose

    pose = hub_sample_pose(model, "7scenes")

    expected = _expected("7scenes")
    assert_pose_close(pose, expected, "7Scenes demo sample")


# ---------------------------------------------------------------------------
# Benchmark set (Hub regression set, or VPT_REGRESSION_DATA)
# ---------------------------------------------------------------------------

FIXTURES = regression_fixture_dirs()
FIXTURE_PARAMS = [pytest.param(d, id=f"{d.parent.name}/{d.name}") for d in FIXTURES] or [
    pytest.param(None, marks=pytest.mark.skip(reason="benchmark regression set unavailable (no Hub access, VPT_REGRESSION_DATA not set)"), id="no-fixtures")]


@pytest.fixture(scope="module")
def benchmark_results(model):
    """Each fixture is run once and shared by the pose and accuracy tests."""
    cache = {}

    def get(d):
        if d not in cache:
            cache[d] = regression_fixture_result(model, d)
        return cache[d]
    return get


def _expected_benchmark(d) -> dict:
    path = EXPECTED / d.parent.name / f"{d.name}.json"
    if not path.exists():
        pytest.skip(f"No recorded output for {d.parent.name}/{d.name}.  "
                    f"Run  python tests/record_regression.py  to generate it.")
    return json.loads(path.read_text())


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_benchmark_pose(benchmark_results, fixture_dir):
    expected = _expected_benchmark(fixture_dir)
    assert_pose_close(benchmark_results(fixture_dir)["T_w2c"], expected["T_w2c"],
                      f"{fixture_dir.parent.name}/{fixture_dir.name}")


@pytest.mark.parametrize("fixture_dir", FIXTURE_PARAMS)
def test_benchmark_accuracy(benchmark_results, fixture_dir):
    expected = _expected_benchmark(fixture_dir)
    got = benchmark_results(fixture_dir)
    assert got["rre_deg"] <= expected["rre_deg"] + RRE_MARGIN_DEG, \
        f"RRE {got['rre_deg']:.3f} deg, recorded {expected['rre_deg']:.3f} deg"
    assert got["rte_m"] <= expected["rte_m"] + RTE_MARGIN_M, \
        f"RTE {got['rte_m']:.3f} m, recorded {expected['rte_m']:.3f} m"
