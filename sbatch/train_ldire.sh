#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time 04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=40G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8g

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

# Define the list of datasets
datasets=("5_steps_split" "10_steps_split" "20_steps_split" "30_steps_split")

# Loop over the datasets
for dataset in "${datasets[@]}"; do
    echo "Training model resnet50_pixel on dataset $dataset"

    COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/training_data/ldire/$dataset.tar"
    # Rsync the dataset to the current directory
    rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR
    tar xf $TMPDIR/$dataset.tar -C $TMPDIR

    # Train the model
    NAME="LDIRE 10k resnet50_latent $dataset"
    MODEL="resnet50_latent"
    DATA_TYPE="latent" # images or latent
    DATA="$TMPDIR/$dataset"
    ls $TMPDIR
    ls $DATA
    python src/training.py \
    --name "$NAME" \
    --model $MODEL \
    --data_type $DATA_TYPE \
    --data_dir $DATA \
    --freeze_backbone 1\
    --batch_size 256 \
    --max_epochs 1000 \

    rm -rf $TMPDIR/dataset


done
