#!/bin/bash

./phenobase/ensemble_vote.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --glob 'data/inference/raw/flower_inference_3?.csv' \
  --glob 'data/inference/raw/flower_inference_2?.csv' \
  --glob 'data/inference/raw/flower_inference_1?.csv' \
  --vote-csv data/inference/flower_inference_2025-08-28.csv
