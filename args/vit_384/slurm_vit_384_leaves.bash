#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=vit_384_leaves

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

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/train_vit.py \
  --output-dir /blue/guralnick/rafe.lafrance/phenobase/data/tuned/vit_384_leaves \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/images_384 \
  --trait-csv /blue/guralnick/rafe.lafrance/phenobase/data/split_all_3.csv \
  --finetune "google/vit-base-patch16-384" \
  --image-size 384 \
  --epochs 200 \
  --lr 1e-4 \
  --batch-size 128 \
  --trait leaves

date