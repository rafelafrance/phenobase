#!/bin/bash

./phenobase/ensemble_inferred_sample.py \
  --winners-csv data/inference/best_3combo_fract/flower_inference_fract_15000_40000_votes3.csv \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_inference_fract3 \
  --image-bash args/slurm/move_flower_ensemble_fract_images.bash \
  --output-sample data/inference/best_3combo_fract/flower_ensemble_fract_sample3.csv \
  --sample-positive 400
