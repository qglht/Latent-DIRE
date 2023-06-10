#!/bin/bash

#SBATCH --job-name=eval_grid
#SBATCH --time 04:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --tmp=20G
#SBATCH --gres=gpumem:24g

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion

# Define the grid of models and datasets
datasets=("5_steps_split" "10_steps_split" "20_steps_split" "30_steps_split")
models=("DIRE 10k resnet50_pixel 5_steps_split" "DIRE 10k resnet50_pixel 10_steps_split" "DIRE 10k resnet50_pixel 20_steps_split" "DIRE 10k resnet50_pixel 30_steps_split")

# Loop over the datasets
for dataset in "${datasets[@]}"; do
    # Loop over the models
    for model in "${models[@]}"; do
        echo "Evaluating model $model on dataset $dataset"

        COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/training_data/dire_split/$dataset.tar"
        # Rsync the dataset to the current directory
        rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR
        tar xf $TMPDIR/$dataset.tar -C $TMPDIR

        # Evaluate the model
        NAME="Eval $model on $dataset"
        MODEL="resnet50_pixel"
        DATA_TYPE="images" # images or latent
        DATA="$TMPDIR/$dataset/test"
        # Find the first checkpoint in the model folder
        CKPT=$(find . -type f -name "epoch=*" | sort -n -t "=" -k 2 | head -n 1)
        python src/eval.py \
        --name "$NAME" \
        --model $MODEL \
        --type $DATA_TYPE \
        --data_dir $DATA \
        --ckpt $CKPT \
        --batch_size 250 \

        rm -rf $TMPDIR/dataset
    done
done