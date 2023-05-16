#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time 04:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32G

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

DATA="$HOME/Latent-DIRE/data/glide"
CKPT="$HOME/Latent-DIRE/models/model.ckpt"
NAME="GLIDE DIRE10 ResNet50"
python src/eval.py --name "$NAME" --data_dir $DATA --batch_size 256                                                             
                                                                                                                                                           