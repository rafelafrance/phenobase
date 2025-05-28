#!/bin/bash

./phenobase/ensemble_vote.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --score-csv data/custom/flower_redbud_inference_1.csv \
  --score-csv data/custom/flower_redbud_inference_2.csv \
  --score-csv data/custom/flower_redbud_inference_3.csv \
  --output-csv data/custom/flower_redbud_ensemble_votes.csv
