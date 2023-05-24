#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time 04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=40G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/ldire_data.tar"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/images.tar
mkdir -p $TMPDIR/images
tar xf $TMPDIR/images.tar -C $TMPDIR/images

DATA="$TMPDIR/images"
NAME="lDIRE_10k_ResNet50_10_steps"
python src/training.py --name $NAME --data_dir $DATA --batch_size 256 --max_epochs 1000 --model "resnet50_latent" --data_type "latent" --num_workers 0 --freeze_backbone 0
                                                                                                                                                           
