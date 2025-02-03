#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_family_score.py \
  --score-csv data/score_single_label_2025-01-28c.csv \
  --family-csv data/families_flowers_2025-02-03b.csv \
  --checkpoint data/tuned/vit_384_lg_flowers_f1/checkpoint-3270 \
  --trait flowers
