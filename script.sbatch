#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --time 24:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpumem:32g
#SBATCH --gpus=1

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
python src/generate_fake.py