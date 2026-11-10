#!/bin/bash
# Multi-node DRS VPT training under Slurm: one torchrun per node, all GPUs on each node.
# The released model was trained on 16 GPUs with max_num_of_imgs_per_gpu=15; set --nodes and --gpus-per-node to match.
#
#   sbatch scripts/slurm_train.sh dataset=drs_vpt train_params.max_num_of_imgs_per_gpu=15
#
# Extra arguments are passed to scripts/train.py as Hydra overrides. Re-submitting with the same
# hydra.run.dir resumes from its checkpoint-last.pth.
#SBATCH --job-name=drs_vpt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

# Load your cluster's CUDA / NCCL modules and activate the environment here, e.g.
#   module load cuda/12.6
#   conda activate drs_vpt

cd "${SLURM_SUBMIT_DIR}"

export MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29600}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1

echo "[$(date)] ${SLURM_NNODES} nodes x ${SLURM_GPUS_ON_NODE} GPUs, master ${MASTER_ADDR}:${MASTER_PORT}"

srun --ntasks-per-node=1 \
  torchrun \
    --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    --nnodes="${SLURM_NNODES}" \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    scripts/train.py \
      dataset.num_workers=12 \
      "$@"

echo "[$(date)] Job finished"
