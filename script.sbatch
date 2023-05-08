#!/bin/bash

#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --time 00:05:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=job-name

module load eth_proxy
python src/generate_fake.py