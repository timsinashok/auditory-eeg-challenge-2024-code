#!/bin/bash
#SBATCH --job-name=dilated
#SBATCH --time=30:00:00
#SBATCH --mem=40000
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mail-type=END
#SBATCH --mail-user=<NetID>@nyu.edu
#SBATCH --output=/scratch/at5282/dilated_lstm4_aug_test.out

# Activate conda environment
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
module load cuda
conda activate tf-gpu
# Script to Execute
python3 /scratch/at5282/code/task1_match_mismatch/experiments/dilated_lstm.py
