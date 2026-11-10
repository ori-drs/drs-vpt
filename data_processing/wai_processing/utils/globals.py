# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), data_processing/wai_processing/utils/globals.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.

from pathlib import Path

WAI_PROC_MAIN_PATH = Path(__file__).parent.parent.parent
WAI_PROC_CONFIG_PATH = WAI_PROC_MAIN_PATH / "wai_processing" / "configs"
WAI_PROC_SCRIPT_PATH = WAI_PROC_MAIN_PATH / "wai_processing" / "scripts"
WAI_SPOD_RUNS_PATH = "/fsx/xrtech/code/spod_runs"
