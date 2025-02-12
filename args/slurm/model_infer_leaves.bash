#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=infer_leaves_0_1000000

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=8:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/model_infer.py \
  --db /home/rafe.lafrance/blue/phenobase/data/gbif_2024-10-28.sqlite \
  --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
  --output-csv /home/rafe.lafrance/blue/phenobase/data/leave_inference_0_1000000.csv \
  --checkpoint /blue/guralnick/rafe.lafrance/phenobase/data/tuned/effnet_528_leaves_f1/checkpoint-2800 \
  --image-size 528 \
  --limit 10000000 \
  --offset 0 \
  --trait leaves \
  --thresh-low 0.10 \
  --thresh-high 0.80

date
