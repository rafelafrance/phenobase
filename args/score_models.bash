#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/score_model.py \
  --trait-csv splits/split_all_3.csv \
  --image-dir data/images/images_384 \
  --output-csv data/score.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits flowers \
  --model-dir data/tuned/vit_384_lg_flowers_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/split_all_3.csv \
  --image-dir data/images/images_384 \
  --output-csv data/score.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits fruits \
  --model-dir data/tuned/vit_384_lg_fruits_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/split_all_3.csv \
  --image-dir data/images/images_384 \
  --output-csv data/score.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits leaves \
  --model-dir data/tuned/vit_384_lg_leaves_prec_wt
