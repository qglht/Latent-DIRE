#!/bin/bash

#SBATCH --gpus=a100-pcie-40gb
#SBATCH --gpus=1
#SBATCH --time 10:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=compute-dire
#SBATCH --gres=gpumem:32g
#SBATCH --tmp=20G

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
rsync -chavzP  /cluster/scratch/hehlif/imagenet.tar $TMPDIR/imagenet.tar
tar xf $TMPDIR/imagenet.tar --directory=$TMPDIR/imagenet
python src/generate_dire.py -d --batch_size 20 --read_dir $TMPDIR/imagenet --write_dir_dire "/cluster/scratch/hehlif/imagenet_dire" --write_dir_latent_dire "/cluster/scratch/hehlif/imagenet_latent_dire"