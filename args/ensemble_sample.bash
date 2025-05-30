#!/bin/bash

./phenobase/ensemble_sample.py \
  --winners-csv data/custom/flower_redbud_ensemble_votes.csv \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_redbud \
  --image-bash args/slurm/move_flower_ensemble_redbud_images.bash \
  --output-sample data/custom/flower_redbud_ensemble_sample3.csv \
  --sample-positive 100
