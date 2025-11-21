#!/bin/bash

#SBATCH --job-name=ad_hoc_ufit_utility

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=4-00:00:00

date
hostname
pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/ad_hoc_ufit_utility.py \
    --archive-csv /home/rafe.lafrance/blue/phenobase/data/flower_inference_formatted_2025-10-21.csv \
    --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
    --output-dir /home/rafe.lafrance/blue/phenobase/data/small_image_archive

date
