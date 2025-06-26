#!/bin/bash

./phenobase/ensemble_sample.py \
  --winners-csv data/audit/flower_more_votes_2025-06-26.csv \
  --image-dir /blue/guralnick/rafe.lafrance/phenobase/data/images/flower_more_votes_2025-06-26 \
  --image-bash args/slurm/flower_more_votes_2025-06-26.bash \
  --output-sample data/audit/flower_more_votes_2025-06-26.csv \
  --sample-positive 250 \
  --sample-negative 100
