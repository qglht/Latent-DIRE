#!/bin/bash

#SBATCH --job-name=compute-dire
#SBATCH --time 24:00:00
#SBATCH --tmp=20G
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g

SAMPLE_FLAGS="--batch_size 100 --num_samples 50000 --timestep_respacing 250"

module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install . src/guided-diffusion
wget --output-document models/256x256_classifier.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt
wget --output-document models/256x256_diffusion.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
cd src/guided-diffusion/scripts/
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python classifier_sample.py $SAMPLE_FLAGS $MODEL_FLAGS --classifier_scale 1.0 --classifier_path models/256x256_classifier.pt --model_path models/256x256_diffusion.pt