#!/bin/bash

#SBATCH --job-name=decode-ldire
#SBATCH --time 04:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/dire/decode_temp.tar/"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/images.tar
mkdir -p $TMPDIR/images
tar xf $TMPDIR/images.tar -C $TMPDIR/images 
python scripts/decode_ldire.py --batch_size 256 --write_dir "/cluster/scratch/$USER/decode_ldire_10k_10st" 