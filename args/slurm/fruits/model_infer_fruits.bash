#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=infer_fruits_0_10000

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
    --bad-families /home/rafe.lafrance/blue/phenobase/datasets/bad_families/bad_flower_fams.csv \
    --output-csv /home/rafe.lafrance/blue/phenobase/data/infer/flower_inference_0_10000_2025-04-12.csv \
    --checkpoint /blue/guralnick/rafe.lafrance/phenobase/data/models/effnet_528_fruits_f1_sl/checkpoint-15260 \
    --image-size 528 \
    --limit 100000 \
    --offset 0 \
    --trait fruits

date
