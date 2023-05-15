#!/bin/bash

#SBATCH --job-name=compute-dire
#SBATCH --time 12:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=a100_80gb:1

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/imagenet.tar"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/images.tar
mkdir -p $TMPDIR/images
tar xf $TMPDIR/images.tar -C $TMPDIR/images 
python scripts/generate_dire.py -d --ddim_steps 10 --batch_size 32 --write_dir_dire "/cluster/scratch/$USER/imagenet_dire" --write_dir_latent_dire "/cluster/scratch/$USER/imagenet_latent_dire"
