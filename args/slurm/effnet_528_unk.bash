#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=effnet_528_flowers_unk_f1

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=4
#SBATCH --mem=4gb
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_train.py \
  --output-dir /blue/guralnick/rafe.lafrance/phenobase/data/models/effnet_528_flowers_unk_f1 \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/phenobase \
  --dataset-csv /blue/guralnick/rafe.lafrance/phenobase/datasets/train_data.csv \
  --finetune "google/efficientnet-b6" \
  --image-size 528 \
  --epochs 200 \
  --trait flowers \
  --problem-type regression \
  --best-metric f1 \
  --use-unknowns \
  --batch-size 32

date
