#!/bin/bash

# Batch job for training the DIRE model on the cluster.
# We train the pixel resnet model for four different datasets,
# each differing in the number of ddim steps.

#SBATCH --job-name=train_batch
#SBATCH --time 4:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8g

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

# Define the list of datasets
datasets=("5_steps" "10_steps" "20_steps" "30_steps")

# Loop over the datasets
for dataset in "${datasets[@]}"; do
    echo "Training model resnet50_pixel on dataset $dataset"

    COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/training_data/dldire/$dataset.tar"
    # Rsync the dataset to the current directory
    rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR
    tar xf $TMPDIR/$dataset.tar -C $TMPDIR

    # Train the model
    NAME="DLDIRE 10k resnet50_pixel $dataset"
    MODEL="resnet50_pixel"
    DATA_TYPE="images" # images or latent
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
