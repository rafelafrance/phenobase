#!/bin/bash

./phenobase/ensemble_sample.py \
  --winners-csv data/custom/flower_redbud_ensemble_votes.csv \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample \
  --image-bash args/slurm/move_flower_ensemble_neg_sample.bash \
  --output-sample data/custom/flower_neg_sample.csv \
  --sample-negative 500
