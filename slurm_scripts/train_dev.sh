#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time 01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=train_dev

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
python src/training.py --data_dir "data/data_dev" --batch_size 10                                                                   
~                                                                                                                                                             
