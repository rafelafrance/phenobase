#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=vit_fruits

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/train_model.py \
  --output-dir /blue/guralnick/rafe.lafrance/phenobase/data/training_output/vit_fruits \
  --pretrained-dir /blue/guralnick/rafe.lafrance/phenobase/data/pretraining_output \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/images_224 \
  --trait-csv /blue/guralnick/rafe.lafrance/phenobase/data/splits.csv \
  --lr 1e-4 \
  --epochs 1200 \
  --batch-size 128 \
  --trait fruits

date
