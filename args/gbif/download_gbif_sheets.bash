#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=download_gbif_sheets

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=16:00:00

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/download_gbif_sheets.py \
  --gbif-db /blue/guralnick/rafe.lafrance/phenobase/data/gbif_2024-10-23.sqlite \
  --image-dir /blue/guralnick/share/phenobase_specimen_data/images/cache_0002 \
  --limit 10000 \
  --offset 10000

date
