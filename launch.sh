#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=12:00:00

python train.py --gin_path "configs/tot.gin" --save_path "runs/tot"
