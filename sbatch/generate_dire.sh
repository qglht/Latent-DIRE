#!/bin/bash

#SBATCH --job-name=compute-dire
#SBATCH --time 12:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/sd_generated.tar.gz"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/images.tar
mkdir -p $TMPDIR/images
tar xf $TMPDIR/images.tar -C $TMPDIR/images 
# make sure you have a few GB of space in your home directory as we save ldire as pt files there 
# you can check with lquota
python scripts/generate_dire.py --ddim_steps 20 --batch_size 10 \
--write_dir_dire "/cluster/scratch/$USER/sd_dire_20" \
--write_dir_ldire "/cluster/scratch/$USER/sd_ldire_20" \
--write_dir_decoded_ldire "/cluster/scratch/$USER/sd_dldire_20"
