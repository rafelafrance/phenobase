#!/bin/bash

#SBATCH --job-name=flower_inference_fract_15000_40000_1

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
    --bad-taxa /home/rafe.lafrance/blue/phenobase/datasets/remove_flowers.csv \
    --output-csv /home/rafe.lafrance/blue/phenobase/data/infer/flower_inference_fract_15000_40000_1_2025-05-19.csv \
    --checkpoint /blue/guralnick/rafe.lafrance/phenobase/data/models/best_3combo_fract/effnet_528_flowers_reg_f1_a_checkpoint-17424 \
    --image-size 528 \
    --limit 15000 \
    --offset 40000 \
    --problem-type regression \
    --trait flowers

date
