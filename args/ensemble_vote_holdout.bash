#!/bin/bash

./phenobase/ensemble_vote.py \
  --ensemble-json data/scores/best_3combo_fract.json \
  --glob 'data/scores/vit_384_lg_flowers_f1_slurm_sl-checkpoint-9450.csv' \
  --glob 'data/scores/vit_384_lg_flowers_f1_a-checkpoint-19199.csv' \
  --glob 'data/scores/effnet_528_flowers_reg_f1_a-checkpoint-17424.csv' \
  --vote-csv data/inference/ensemble_votes_holdout_2025-10-23.csv
