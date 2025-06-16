#!/bin/bash

./phenobase/ensemble_vote.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --score-csv data/inference/raw/flower_inference_1000000_0000000_1_2025-06-11.csv \
  --score-csv data/inference/raw/flower_inference_1000000_0000000_2_2025-06-11.csv \
  --score-csv data/inference/raw/flower_inference_1000000_0000000_3_2025-06-11.csv\
  --vote-csv data/inference/flower_inference_1000000_0000000_votes.csv
