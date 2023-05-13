#!bin/bash

#SBATCH --gpus=a100-pcie-40gb
#SBATCH --gpus=1
#SBATCH --time 16:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=job-name
#SBATCH --gres=gpumem:32g

module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9
python src/generate_fake.py