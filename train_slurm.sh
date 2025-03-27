#!/bin/bash
#SBATCH --job-name=lolcats_train
#SBATCH --output=lolcats_train_%j.out
#SBATCH --error=lolcats_train_%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

bash setup.bash
bash train.bash