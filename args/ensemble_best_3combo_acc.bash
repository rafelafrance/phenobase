#!/bin/bash

./phenobase/ensemble_inferred_audit.py \
  --ensemble-json data/scores/best_3combo_acc.json \
  --score-csv data/inference/best_3combo_acc/flower_inference_10000_20000_2_2025-05-08.csv \
  --score-csv data/inference/best_3combo_acc/flower_inference_10000_20000_3_2025-05-08.csv \
  --score-csv data/inference/best_3combo_acc/flower_inference_10000_20000_1_2025-05-08.csv \
  --output-csv data/inference/best_3combo_acc/flower_inference_acc_10000_20000_votes.csv
