#!/bin/bash

#SBATCH --partition=norm-gpu --gres=gpu:2 --output=mae_ddp.out
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -t 00:30:00
#SBATCH --mem=12g

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1) 
export MASTER_PORT=12345 

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate mae

# Run program
cd /users/bjoo2/code/mae_cs2952x

echo "Pretraining MAE"
srun torchrun \
        --nnodes=2 \
        --nproc_per_node=1 \
        --rdzv_id=100 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:29400 \
         ddp_pretrain.py --batch_size 64 --model large --epochs 2


