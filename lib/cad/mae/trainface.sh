#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J digitalshadow
#SBATCH -o digitalshadow.%J.out
#SBATCH -e digitalshadow.%J.err
#SBATCH --mail-user=zhongyi.han@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=14-00:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:8  # Requesting GPUs
#SBATCH --cpus-per-task=16  # Adjusted for GPUs
#SBATCH --constraint=v100


python -m torch.distributed.launch --nproc_per_node=8 main_pretrain_face.py \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path /ibex/scratch/projects/c2108/juexiao/ \
    --output_dir ./output_face