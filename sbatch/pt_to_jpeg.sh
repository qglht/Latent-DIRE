#!/bin/bash

#SBATCH --job-name=pt-to-jpeg
#SBATCH --time 04:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=4G

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

PT_BATCHES_PATH="/cluster/scratch/$USER/imagenet_dire"
python scripts/pt_to_jpeg.py --read_dir $PT_BATCHES_PATH --write_dir "$HOME/data/processed/imagenet_dire_jpeg"