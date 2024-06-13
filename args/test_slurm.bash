#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=mae_224_pretraining

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date;hostname;pwd

module purge all

# module load ubuntu/22.04
module load ngc-pytorch/2.3.0
# module load conda

uname -a

# mamba activate /blue/guralnick/rafe.lafrance/.conda/envs/vitmae/
export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/test_slurm.py

date
