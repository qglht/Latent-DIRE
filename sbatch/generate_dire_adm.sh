#!/bin/bash

#SBATCH --job-name=compute-adm-dire
#SBATCH --time 12:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

COMPRESSED_FOLDER_PATH="/cluster/scratch/$USER/sd_generated.tar.gz"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
wandb login
pip install . src/guided-diffusion
rsync -chavzP $COMPRESSED_FOLDER_PATH $TMPDIR/images.tar
mkdir -p $TMPDIR/images
tar xf $TMPDIR/images.tar -C $TMPDIR/images 
python scripts/generate_dire.py --model_id "models/lsun_bedroom.pt" --ddim_steps 20 --batch_size 10 \
--write_dir_dire "/cluster/scratch/$USER/sd_admdire_20" \
