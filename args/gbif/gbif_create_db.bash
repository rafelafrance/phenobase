#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=gbif_create_db

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

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/gbif_create_db.py \
  --multimedia-tsv /blue/guralnick/rafe.lafrance/phenobase/data/gbif_metadata/multimedia.txt \
  --occurrence-tsv /blue/guralnick/rafe.lafrance/phenobase/data/gbif_metadata/occurrence.txt \
  --gbif-db /blue/guralnick/rafe.lafrance/phenobase/data/gbif_2024-10-28.sqlite

date
