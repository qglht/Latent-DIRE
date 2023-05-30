#!/bin/bash

#SBATCH --job-name=train_batch
#SBATCH --time 12:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

# Define the list of datasets
datasets=("dire_5" "dire_10" "dire_20" "dire_30")

# Loop over the datasets
for dataset in "${datasets[@]}"; do
    echo "Training model resnet50_pixel on dataset $dataset"

    COMPRESSED_FOLDER_PATH = "/cluster/scratch/$USER/training_data/$dataset.tar"
    # Rsync the dataset to the current directory
    rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/dataset.tar
    mkdir -p $TMPDIR/dataset
    tar xf $TMPDIR/dataset.tar -C $TMPDIR/dataset

    # Train the model
    NAME="DIRE 10k resnet50_pixel"
    MODEL="resnet50_pixel"
    DATA_TYPE="images" # images or latent
    DATA="$TMPDIR/dataset"
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