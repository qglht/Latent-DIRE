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
models=("resnet_50_pixel_5_steps" "resnet_50_pixel_10_steps" "resnet_50_pixel_20_steps" "resnet_50_pixel_30_steps")

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
        DATA_TYPE="images" # images or latent
        DATA="$TMPDIR/$dataset/test"
        MODEL="resnet50_pixel"

        # Find the first checkpoint in the model folder
        checkpoint_dir="/cluster/home/$USER/Latent-DIRE/models/dire/$model"
        # Get the list of files in the directory
        files=("$checkpoint_dir"/*)

        # Check if there is only one file in the directory
        if [ ${#files[@]} -eq 1 ]; then
            # Return the path to the file
            CKPT="${files[0]}"
            echo "$CKPT"
        else
            # Handle the case where there are multiple files or no files
            echo "Error: There is not exactly one file in the directory."
        fi
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