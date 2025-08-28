#!/bin/bash

#SBATCH --job-name=3j_flower_inference

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=7-00:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

date
hostname
pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_infer_custom.py \
    --custom-data /home/rafe.lafrance/blue/phenobase/data/not_inferred.csv \
    --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
    --bad-taxa /home/rafe.lafrance/blue/phenobase/datasets/remove_flowers.csv \
    --output-csv /home/rafe.lafrance/blue/phenobase/data/infer/flower_inference_3j.csv \
    --checkpoint /blue/guralnick/rafe.lafrance/phenobase/data/models/best_3combo_fract/vit_384_lg_flowers_f1_slurm_sl_checkpoint-9450 \
    --image-size 384 \
    --trait flowers
