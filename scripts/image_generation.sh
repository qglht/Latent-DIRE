#!/bin/bash

#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --time 05:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=job-name

module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9
pip install -U torch torchvision torchaudio
pip install -U xformers
pip install -U diffusers["torch"] accelerate transformers
python src/generate_fake.py