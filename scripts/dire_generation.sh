#!/bin/bash

#SBATCH --job-name=compute-dire
#SBATCH --time 12:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/imagenet.tar.gz"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/images.tar.gz
mkdir -p $TMPDIR/images
tar xzf $TMPDIR/images.tar.gz -C $TMPDIR/images 
python src/generate_dire.py --ddim_steps 10 --batch_size 20 --write_dir_dire "/cluster/scratch/$USER/imagenet_dire" --write_dir_latent_dire "/cluster/scratch/$USER/imagenet_latent_dire"
