#!/bin/bash

./phenobase/ensemble_inferred_audit.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --score-csv data/inference/best_3combo_fract/flower_inference_fract_10000_30000_3_2025-05-09.csv \
  --score-csv data/inference/best_3combo_fract/flower_inference_fract_10000_30000_2_2025-05-09.csv \
  --score-csv data/inference/best_3combo_fract/flower_inference_fract_10000_30000_1_2025-05-09.csv \
  --output-csv data/inference/best_3combo_fract/flower_inference_fract_10000_30000_votes2.csv
