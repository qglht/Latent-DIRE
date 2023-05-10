#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time 01:00:00
#SBATCH --ntasks_per_node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=train_dev

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
python src/training.py --train_dir "data/dev_train" --val_dir "data/dev_val" --test_dir "data/dev_test" --batch_size 10                                                                   
~                                                                                                                                                             
