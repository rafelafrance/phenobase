#!/bin/bash

#SBATCH --job-name=flower_inference_10000_20000_2

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=24:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date
hostname
pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_infer.py \
    --db /home/rafe.lafrance/blue/phenobase/data/gbif_2024-10-28.sqlite \
    --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
    --output-csv /home/rafe.lafrance/blue/phenobase/data/infer/flower_inference_10000_20000_2_2025-05-08.csv \
    --checkpoint /blue/guralnick/rafe.lafrance/phenobase/data/models/best_3combo_acc/effnet_528_flowers_unk_f1_a_checkpoint-28000 \
    --image-size 528 \
    --limit 10000 \
    --offset 20000 \
    --problem-type regression \
    --bad-taxa /home/rafe.lafrance/blue/phenobase/datasets/remove_flowers.csv \
    --trait flowers

date
