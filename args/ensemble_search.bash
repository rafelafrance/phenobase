#!/bin/bash

./phenobase/ensemble_search.py \
  --checkpoint-limit 0.85 \
  --checkpoint-metric "accuracy" \
  --combo-filter "accuracy" \
  --combo-limit 0.9 \
  --combo-order "fraction" \
  --checkpoint-limit 0.85 \
  --metric-csv "data/scores/flower_test_metrics.csv" \
  --output-json "data/scores/best_3combo_fract-2025-10-23.json" \
  --score-csv "data/scores/all_flower_test_scores.csv" \
  --vote-threshold 2 \
  --votes 3
