#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=infer_flowers

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_infer.py \
  --db /home/rafe.lafrance/blue/phenobase/data/gbif_2024-10-28.sqlite \
  --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
  --bad-families /home/rafe.lafrance/blue/phenobase/datasets/bad_families/bad_flower_fams.csv \
  --output-csv /home/rafe.lafrance/blue/phenobase/data/flower_inference_0_10.csv \
  --model-dir /blue/guralnick/rafe.lafrance/phenobase/data/tuned/vit_384_lg_flowers_f1/checkpoint-3270 \ \
  --image-size 384 \
  --limit 10 \
  --offset 0

date
