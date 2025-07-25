#!/bin/bash

#SBATCH --job-name=effnet_600_flowers_f1_unk_51

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
  --output-dir /blue/guralnick/rafe.lafrance/phenobase/data/models/effnet_600_flowers_f1_unk_51 \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/phenobase \
  --dataset-csv /blue/guralnick/rafe.lafrance/phenobase/datasets/splits_2025-04-22.csv \
  --finetune "google/efficientnet-b7" \
  --image-size 600 \
  --epochs 100 \
  --trait flowers \
  --best-metric f1 \
  --use-unknowns \
  --problem-type regression \
  --batch-size 16

date
