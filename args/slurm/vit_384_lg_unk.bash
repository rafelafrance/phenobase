#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=vit_384_lg_unk_flowers

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_train.py \
  --output-dir /blue/guralnick/rafe.lafrance/phenobase/data/models/vit_384_lg_flowers \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/phenobase \
  --dataset-csv /blue/guralnick/rafe.lafrance/phenobase/datasets/all_traits.csv \
  --finetune "google/vit-large-patch16-384" \
  --image-size 384 \
  --epochs 200 \
  --trait flowers \
  --use-unknowns \
  --batch-size 16

date
