# Adapted from MapAnything (https://github.com/facebookresearch/map-anything), scripts/train.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. Apache License 2.0, see drs_vpt/_vendor/mapanything/LICENSE.

"""
Train or fine-tune DRS VPT. Configuration is managed by Hydra (configs/train.yaml).

Usage:
    bash bash_scripts/train/examples/drsvpt_kitti_4v_15ipg.sh <NUM_GPUS>   # KITTI only
    bash bash_scripts/train/main/drsvpt_11d_4v_15ipg.sh <NUM_GPUS>         # all datasets (released mix)
    # several nodes under Slurm: see scripts/slurm_train.sh
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from drs_vpt.train.loop import train


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def execute_training(cfg: DictConfig):
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    execute_training()
