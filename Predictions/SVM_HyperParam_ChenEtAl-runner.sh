#!/bin/bash
#
#SBATCH --job-name=SVMhyper
#SBATCH --time=12:00:00
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/%A.out
#SBATCH --error=outputs/%A.err

module purge
module load python3/intel/3.5.3
module load nltk/python3.5/3.2.4

python3 -u SVM_HyperParam_ChenEtAl.py $1 --main_data_dir "/scratch/eff254/Optimization/Data/"