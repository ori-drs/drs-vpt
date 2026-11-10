# DRS VPT Training

## Setup

Install the training extra on top of the install in `README.md`:

```bash
pip install -e ".[train]"
```

Prepare the training data in WAI format following `data_processing/README.md`, then set the data paths in
`configs/machine/local.yaml` (or add `configs/machine/<your_machine>.yaml` and pass `machine=<your_machine>`).

## Quick Start: Single Dataset Training

Fine-tune the released weights on KITTI odometry only:

```bash
bash bash_scripts/train/examples/drsvpt_kitti_4v_15ipg.sh 1    # number of GPUs
```

**Tips:**

- `train_params.max_num_of_imgs_per_gpu=15` uses 89 GB per GPU at 518 px; lower it to fit smaller GPUs.
- Scale the learning rate with the effective batch size (at 4 views: `NUM_GPUS × max_num_of_imgs_per_gpu / 4`).
- Keep the warmup when fine-tuning: training starts with a fresh optimizer.
- `model.pretrained` takes the Hub weights (`fulkast/drs_vpt`), a training checkpoint (`.pth`), or `null` to start
  from pretrained DINOv2 and Sonata.
- Extra Hydra overrides after the number of GPUs are passed to `scripts/train.py`.

## Main Training Script

Fine-tune on all 11 training datasets with the mix the released model was trained on:

```bash
bash bash_scripts/train/main/drsvpt_11d_4v_15ipg.sh 8
```

For several nodes under Slurm, use `sbatch scripts/slurm_train.sh` (it passes its arguments to `scripts/train.py`).

## Configuration System

DRS VPT uses Hydra for configuration management:

- **Overall**: `configs/train.yaml` - Entrypoint
- **Machine**: `configs/machine/` - Path settings
- **Dataset**: `configs/dataset/` - Data mixes (`drs_vpt.yaml`, `kitti.yaml`), shared settings (`default.yaml`) and
  each dataset's settings (`<name>_wai/`: point cloud construction, covisibility threshold, augmentation)
- **Model**: `configs/model/` - Model architecture
- **Loss**: `configs/loss/` - Training and validation criteria
- **Train Params**: `configs/train_params/` - Training hyperparameters

A data mix lists the number of samples per epoch from each dataset. To train on your own mix, add a file to
`configs/dataset/` and pass `dataset=<file name>`:

```yaml
# configs/dataset/kitti_nuscenes.yaml
defaults:
  - default

train_dataset:
  "+ 10_000 @ ${dataset.kitti_wai.train.dataset_str}
  + 10_000 @ ${dataset.nuscenes_wai.train.dataset_str}"

test_dataset:
  "+ 1_000 @ ${dataset.kitti_wai.val.dataset_str}
  + 1_000 @ ${dataset.nuscenes_wai.val.dataset_str}"
```

## Code

- `drs_vpt/train/`: the training loop, losses and utilities.
- `drs_vpt/datasets/`: one class per dataset; `loaders.py` builds the data loaders. The MapAnything code they build on
  is in `drs_vpt/_vendor/mapanything/`, with the changes in `datasets.patch`.
- A training batch is `{"image_views": [...], "cloud": {...}}`: a sample's images, and one point cloud accumulated from
  several frames and placed at a random scanner pose.

Results are written to `hydra.run.dir` (set in each script), with regular checkpoints and training logs; re-running a
script resumes from its `checkpoint-last.pth`.
