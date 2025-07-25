#!/bin/bash

#SBATCH --job-name=vit_384_lg_flowers_f1_reg

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_train.py \
  --output-dir /blue/guralnick/rafe.lafrance/phenobase/data/models/vit_384_lg_flowers_f1_reg \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/phenobase \
  --dataset-csv /blue/guralnick/rafe.lafrance/phenobase/datasets/splits_2025-04-22.csv \
  --finetune "google/vit-large-patch16-384" \
  --image-size 384 \
  --batch-size 16 \
  --best-metric f1 \
  --problem-type regression \
  --epochs 100 \
  --trait flowers

date
