#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time 04:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=20G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

DATA="$HOME/Latent-DIRE/data/train"
NAME="DIRE_10k_ResNet50_20_steps"
python src/training.py --name $NAME --data_dir $DATA --batch_size 256 --max_epochs 1000                                                            
                                                                                                                                                           
