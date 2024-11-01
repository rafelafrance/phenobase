#!/bin/bash

#---------------------------------------------------------------------------------------------
./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --model-dir data/tuned/effnet_528_all_prec_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits flowers \
  --model-dir data/tuned/effnet_528_flowers_prec_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits fruits \
  --model-dir data/tuned/effnet_528_fruits_prec_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits leaves \
  --model-dir data/tuned/effnet_528_leaves_prec_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits flowers \
  --model-dir data/tuned/effnet_528_flowers_prec_nowt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits fruits \
  --model-dir data/tuned/effnet_528_fruits_prec_nowt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits leaves \
  --model-dir data/tuned/effnet_528_leaves_prec_nowt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits flowers \
  --model-dir data/tuned/effnet_528_flowers_f1_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits fruits \
  --model-dir data/tuned/effnet_528_fruits_f1_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 528 \
  --batch-size 1 \
  --traits leaves \
  --model-dir data/tuned/effnet_528_leaves_f1_wt

#---------------------------------------------------------------------------------------------
./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 384 \
  --batch-size 1 \
  --model-dir data/tuned/vit_384_lg_all_prec_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits flowers \
  --model-dir data/tuned/vit_384_lg_flowers_prec_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits fruits \
  --model-dir data/tuned/vit_384_lg_fruits_prec_wt

./phenobase/model_score.py \
  --trait-csv splits/filtered_families.csv \
  --image-dir data/images/images_600 \
  --output-csv data/score_filtered.csv \
  --image-size 384 \
  --batch-size 1 \
  --traits leaves \
  --model-dir data/tuned/vit_384_lg_leaves_prec_wt
