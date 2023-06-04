#!/bin/bash

#SBATCH --job-name=train_adm
#SBATCH --time 04:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8g

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion



NAME="ADM DIRE 10k ResNet50"
MODEL="resnet50_pixel" # resnet50_pixel or resnet50_latent
DATA_TYPE="images" # images or latent

dataset="20_steps"

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/training_data/adm_dire/$dataset.tar"
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR
tar xf $TMPDIR/$dataset.tar -C $TMPDIR

DATA="$TMPDIR/$dataset"

python src/training.py \
--name "$NAME" \
--model $MODEL \
--data_type $DATA_TYPE \
--data_dir $DATA \
--freeze_backbone 1 \
--batch_size 256 \
--max_epochs 1000 \
--num_workers 0 # if used, specify #SBATCH --cpus-per-task
                                                                                                                                                           
