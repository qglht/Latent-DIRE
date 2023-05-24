#!/bin/bash

#SBATCH --job-name=pt-to-npz
#SBATCH --time 00:10:00
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=4G

PT_PATH="/cluster/scratch/$USER/imagenet_ldire20"
NPZ_PATH="/cluster/scratch/$USER/imagenet_ldire20_npz"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
python scripts/pt_to_npz.py --read_dir $PT_PATH --write_dir $NPZ_PATH
