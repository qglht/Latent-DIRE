#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time 04:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=20G
#SBATCH --gpus=a100-pcie-40gb:1

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

DATA="$HOME/Latent-DIRE/data/data"
NAME="LDIRE 10k ResNet50"
python src/training.py --name $NAME --data_dir $DATA --batch_size 256 --max_epochs 1000                                                            
                                                                                                                                                           