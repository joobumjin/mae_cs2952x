#!/bin/bash

#SBATCH --partition=gpu --gres=gpu:1 --output=probe.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH --mem=8g

# Load a CUDA module
module load cuda
module load miniconda3/23.11.0s
source /gpfs/runtime/opt/miniconda/4.12.0/etc/profile.d/conda.sh

conda activate mae

# Run program
cd /users/bjoo2/code/mae_cs2952x

echo "FineTuning MAE"
python3 simple_linprobe.py --batch_size 256 --lr 1e-4