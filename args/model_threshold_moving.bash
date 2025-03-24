#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_threshold_moving.py \
  --score-csv data/score_single_label_2025-01-28c.csv \
  --trait fruits \
  --bad-families datasets/bad_families/bad_fruit_fams.csv
