#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time 04:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=20G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

DATA="/cluster/scratch/$USER/data"
tar cf $DATA.tar $DATA
rsync -chavzP $DATA.tar $TMPDIR
tar xf $TMPDIR/$DATA.tar -C $TMPDIR
python src/training.py -d --data_dir "$TMPDIR/data" --batch_size 128 --max-epochs 1000                                                            
                                                                                                                                                           