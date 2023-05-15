#!/bin/bash

#SBATCH --job-name=compute-dire
#SBATCH --time 12:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/imagenet.tar"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/images.tar
mkdir -p $TMPDIR/images
tar xf $TMPDIR/images.tar -C $TMPDIR/images 
# make sure you have a few GB of space in your home directory as we save ldire as pt files there 
# you can check with lquota
python scritps/generate_dire.py --ddim_steps 10 --batch_size 20 \
--write_dir_dire "/cluster/home/$USER/Latent-DIRE/data/imagenet_dire" \
--write_dir_ldire "/cluster/home/$USER/Latent-DIRE/data/imagenet_ldire"
--write_dir_decoded_ldire "/cluster/home/$USER/Latent-DIRE/data/imagenet_decoded_ldire"
