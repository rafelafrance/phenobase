#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=gbif_download_multitasking
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=ALL

#SBATCH --mem=8gb
#SBATCH --nodes=1
#SBATCH --time=10-00:00:00

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/gbif_download_multitasking.py \
  --gbif-db /blue/guralnick/rafe.lafrance/phenobase/data/gbif_2024-10-28.sqlite \
  --image-dir /blue/guralnick/share/phenobase_specimen_data/images \
  --dir-suffix a \
  --offset 0000000 \
  --max-workers 24

date
