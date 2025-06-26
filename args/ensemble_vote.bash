#!/bin/bash

./phenobase/ensemble_vote.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --score-csv data/audit/flower_inference_paper_1a.csv \
  --score-csv data/audit/flower_inference_paper_2a.csv \
  --score-csv data/audit/flower_inference_paper_3a.csv \
  --vote-csv data/audit/flowers_more_votes_2025-06-26.csv
