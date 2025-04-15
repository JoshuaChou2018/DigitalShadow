#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J spinemae
#SBATCH -o spinemae.%J.out
#SBATCH -e spinemae.%J.err
#SBATCH --mail-user=zhongyi.han@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=14-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:8  # Requesting GPUs
#SBATCH --cpus-per-task=8  # Adjusted for GPUs
#SBATCH --constraint=v100
#SBATCH --account conf-icml-2025.01.31-gaox


python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /ibex/scratch/projects/c2108/zhongyi/SpineFoundation/data/ \
    --output_dir ./output
