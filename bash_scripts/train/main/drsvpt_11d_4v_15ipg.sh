#!/bin/bash
# Fine-tune DRS VPT on all 11 training datasets with the released mix (configs/dataset/drs_vpt.yaml).
#   bash bash_scripts/train/main/drsvpt_11d_4v_15ipg.sh <NUM_GPUS> [extra Hydra overrides]
# The released model was trained on 16 GPUs (4 nodes); for several nodes use scripts/slurm_train.sh.

NUM_GPUS=${1:-8}
export HYDRA_FULL_ERROR=1

## max_num_of_imgs_per_gpu=15 uses 89 GB per GPU at 518 px; lower it on smaller GPUs.
## Effective batch size at 4 views = NUM_GPUS x max_num_of_imgs_per_gpu / 4; scale the learning rate with it.
torchrun --nproc_per_node ${NUM_GPUS} \
    scripts/train.py \
    machine=local \
    dataset=drs_vpt dataset.num_workers=12 \
    dataset.num_views=4 \
    model=drs-vpt \
    model.pretrained=fulkast/drs_vpt \
    loss=drs_vpt \
    train_params=drs_vpt \
    train_params.lr=1e-5 \
    train_params.epochs=100 \
    train_params.warmup_epochs=10 \
    train_params.max_num_of_imgs_per_gpu=15 \
    hydra.run.dir='${machine.root_experiments_dir}/drs_vpt/11d_4v_15ipg' \
    "${@:2}"
