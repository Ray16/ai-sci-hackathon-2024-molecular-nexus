#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00

source activate AI4Science_hackathon

cd /project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus

python 1_hyperparameter_tuning.py --out_channels_l1 32 --n_head_l1 8 --out_channels_l2 32 --n_head_l2 8 --learning_rate 0.01 --max_epoch 400