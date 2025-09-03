#!/bin/bash

./phenobase/ensemble_format.py \
  --winners-csv data/inference/flower_inference_votes_2025-09-02.csv \
  --output-csv data/inference/flower_inference_formatted_2025-09-03.csv \
  --gbif-db data/gbif_2024-10-28.sqlite \
  --subset positives_only
