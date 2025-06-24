#!/bin/bash

#SBATCH --job-name=gbif_audit_images2_3

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --time=4:00:00

date
hostname
pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/gbif_audit_images2.py \
    --gbif-db /home/rafe.lafrance/blue/phenobase/data/gbif_2024-10-28.sqlite \
    --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
    --limit 10000000 \
    --offset 30000000

date
