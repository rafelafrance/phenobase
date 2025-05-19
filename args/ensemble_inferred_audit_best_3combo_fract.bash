#!/bin/bash

./phenobase/ensemble_inferred_audit.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --score-csv data/inference/best_3combo_fract/flower_inference_fract_15000_40000_3_2025-05-19.csv \
  --score-csv data/inference/best_3combo_fract/flower_inference_fract_15000_40000_2_2025-05-19.csv \
  --score-csv data/inference/best_3combo_fract/flower_inference_fract_15000_40000_1_2025-05-19.csv \
  --output-csv data/inference/best_3combo_fract/flower_inference_fract_15000_40000_votes3.csv
