#!/bin/bash

####################################################################################################
#---------------------------------------------------------------------------------------------
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --glob-dir data/tuned/*224*

./phenobase/score_model.py \
  --trait-csv data/split_all_3.csv \
  --image-dir data/images/images_384 \
  --output-csv data/score.csv \
  --image-size 384 \
  --trait flowers \
  --glob 'data/tuned/*384*'
