#!/bin/bash

#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --time 10:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=job-name

module load eth_proxy
python src/generate_fake.py