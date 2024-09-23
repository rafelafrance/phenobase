#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 528 \
  --batch-size 1 \
  --model-dir data/tuned/effnet_528_all_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits flowers \
  --model-dir data/tuned/effnet_528_flowers_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits fruits \
  --model-dir data/tuned/effnet_528_fruits_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits leaves \
  --model-dir data/tuned/effnet_528_leaves_prec_wt

#---------------------------------------------------------------------------------------------
./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 384 \
  --batch-size 1 \
  --model-dir data/tuned/vit_384_lg_all_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits flowers \
  --model-dir data/tuned/vit_384_lg_flowers_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits fruits \
  --model-dir data/tuned/vit_384_lg_fruits_prec_wt

./phenobase/score_model.py \
  --trait-csv splits/splits.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits leaves \
  --model-dir data/tuned/vit_384_lg_leaves_prec_wt
