#!/bin/bash

./phenobase/ensemble_vote.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --score-csv data/inference/raw/flower_inference_3c.csv \
  --score-csv data/inference/raw/flower_inference_2c.csv \
  --score-csv data/inference/raw/flower_inference_1c.csv \
  --vote-csv data/inference/flower_inference_2025-07.csv
