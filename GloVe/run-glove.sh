#!/bin/bash
#
#SBATCH --job-name=charrrr
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --mem=40GB
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load nltk/python3.5/3.2.4

## python3 -m nltk.downloader all ## Just once. Now I have the folder in my home dir.
python3 -u GloVe.py --embedding_dim 300 --top_k 10000 --minibatch 10000  --main_data_dir "/scratch/eff254/Optimization/Data/"
