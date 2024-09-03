#!/bin/bash

####################################################################################################
#---------------------------------------------------------------------------------------------
./phenobase/score_model.py \
  --trait-csv data/split_all_3.csv \
  --image-dir data/images/images_224 \
  --output-csv data/score.csv \
  --image-size 224 \
  --model-dir data/tuned/vit_224

./phenobase/score_model.py \
  --trait-csv data/split_all_3.csv \
  --image-dir data/images/images_224 \
  --output-csv data/score.csv \
  --image-size 224 \
  --model-dir data/tuned/vit_224_mae

./phenobase/score_model.py \
  --trait-csv data/split_all_3.csv \
  --image-dir data/images/images_224 \
  --output-csv data/score.csv \
  --image-size 224 \
  --model-dir data/tuned/vit_224_pos

####################################################################################################
#---------------------------------------------------------------------------------------------
./phenobase/score_model.py \
  --trait-csv data/split_all_3.csv \
  --image-dir data/images/images_384 \
  --output-csv data/score.csv \
  --image-size 384 \
  --model-dir data/tuned/vit_384

./phenobase/score_model.py \
  --trait-csv data/split_all_3.csv \
  --image-dir data/images/images_384 \
  --output-csv data/score.csv \
  --image-size 384 \
  --model-dir data/tuned/vit_384_pos
