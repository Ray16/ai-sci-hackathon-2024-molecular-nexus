#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00

source activate AI4Science_hackathon

cd /project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus

python 1_hyperparameter_tuning_FastRGCNNet.py --out_channels_l1 16 --num_relations_l1 64 --out_channels_l2 64 --num_relations_l2 64 --learning_rate 0.005 --max_epoch 800