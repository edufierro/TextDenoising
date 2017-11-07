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
python3 -m pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl --user --upgrade
python3 -m pip install torchvision --user --upgrade

python3 -u GloVe.py --embedding_dim 200 --top_k 10000 --minibatch 10000  --main_data_dir "/scratch/eff254/Optimization/Data/TXTsOriginal/"
