#!/bin/bash

#SBATCH --job-name=flower_inference_10000_10000

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=2
#SBATCH --mem=64gb
#SBATCH --time=24:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

date
hostname
pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_infer.py \
    --db /home/rafe.lafrance/blue/phenobase/data/gbif_2024-10-28.sqlite \
    --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
    --bad-families /home/rafe.lafrance/blue/phenobase/datasets/bad_families/bad_flower_fams.csv \
    --output-csv /home/rafe.lafrance/blue/phenobase/data/infer/flower_inference_10000_10000_2025-04-17.csv \
    --checkpoint /blue/guralnick/rafe.lafrance/phenobase/data/models/effnet_528_flowers_f1_sl/checkpoint-15260 \
    --image-size 528 \
    --limit 10000 \
    --offset 10000 \
    --trait flowers

date
