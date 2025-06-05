#!/bin/bash

./phenobase/ensemble_sample.py \
  --winners-csv data/inference/best_3combo_fract/flower_inference_fract_15000_40000_votes3.csv \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_neg_sample2 \
  --image-bash args/slurm/move_flower_ensemble_neg_sample.bash \
  --output-sample data/inference/best_3combo_fract/flower_neg_sample2.csv \
  --sample-negative 500
