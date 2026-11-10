"""
Generate the expected outputs used by test_regression.py.

Run this once after the model is finalised, then commit what it writes in tests/assets/regression/.
Re-run only when you intentionally change model behaviour (e.g. new checkpoint, architecture change).

KITTI and 7Scenes are recorded from the Hub demo samples, with the Hub model unless VPT_CHECKPOINT
points to a local checkpoint. The benchmark fixture set (the Hub regression set, or VPT_REGRESSION_DATA) is recorded
to <dataset>/<name>.json.

Usage:
    [VPT_CHECKPOINT=/path/to/ckpt.pth] [VPT_DEVICE=cuda] [VPT_REGRESSION_DATA=/path/to/fixtures] \\
    python tests/record_regression.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
ASSETS = Path(__file__).parent / "assets" / "regression"
ASSETS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


def _save(name: str, pose: np.ndarray):
    out = ASSETS / f"{name}_expected_pose.npy"
    np.save(out, pose)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Import shared helpers from conftest without invoking pytest
# ---------------------------------------------------------------------------

from conftest import hub_sample_pose, regression_fixture_dirs, regression_fixture_result  # noqa: E402


def record_hub_sample(model, name: str):
    _save(name, hub_sample_pose(model, name))


def record_fixture(model, d: Path):
    result = regression_fixture_result(model, d)
    out = ASSETS / d.parent.name / f"{d.name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[saved] {out}  RRE {result['rre_deg']:.3f} deg  RTE {result['rte_m']:.3f} m")


if __name__ == "__main__":
    from drs_vpt.hub import load_model

    print("Loading model...")
    model = load_model(os.environ.get("VPT_CHECKPOINT"), device=os.environ.get("VPT_DEVICE", "cpu"))

    for name in ("kitti", "7scenes"):
        print(f"\nRecording {name} fixture...")
        record_hub_sample(model, name)

    if dirs := regression_fixture_dirs():
        print(f"\nRecording {len(dirs)} benchmark fixtures...")
        for d in dirs:
            record_fixture(model, d)
    else:
        print("[skip] benchmark regression set unavailable")

    print("\nDone. Commit the files in tests/assets/regression/")
