#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_threshold_moving.py \
  --score-csv data/score_old_regression_2025_01_29a.csv \
  --problem-type single_label_classification \
  --trait old_flowers
