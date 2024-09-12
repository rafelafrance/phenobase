#!/bin/bash

#---------------------------------------------------------------------------------------------
#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 528 \
#  --batch-size 1 \
#  --model-dir data/tuned/effnet_528_f1 \
#  --model-dir data/tuned/effnet_528_prec \
#  --model-dir data/tuned/effnet_528_f1_wt

#---------------------------------------------------------------------------------------------
#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_224 \
#  --output-csv data/score.csv \
#  --image-size 224 \
#  --model-dir data/tuned/vit_224 \
#  --model-dir data/tuned/vit_224_large_hf \
#  --model-dir data/tuned/vit_224_mae \
#  --model-dir data/tuned/vit_224_pos

#---------------------------------------------------------------------------------------------
#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_384 \
#  --output-csv data/score.csv \
#  --image-size 384 \
#  --batch-size 1 \
#  --model-dir data/tuned/vit_384 \
#  --model-dir data/tuned/vit_384_base_hf \
#  --model-dir data/tuned/vit_384_large_hf_f1 \
#  --model-dir data/tuned/vit_384_pos \
#  --model-dir data/tuned/vit_384_lg_prec

#---------------------------------------------------------------------------------------------
#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 456 \
#  --traits flowers \
#  --model-dir data/local/effnet_456_flowers \
#  --model-dir data/local/effnet_456_flowers_pos

#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 456 \
#  --model-dir data/local/effnet_456_prec \
#  --model-dir data/local/vit_384_base_prec \

#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 528 \
#  --batch-size 1 \
#  --traits flowers \
#  --model-dir data/tuned/effnet_528_flowers_f1_wt

#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 528 \
#  --batch-size 1 \
#  --traits fruits \
#  --model-dir data/tuned/effnet_528_fruits_f1_wt

##./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 528 \
#  --batch-size 1 \
#  --traits leaves \
#  --model-dir data/tuned/effnet_528_leaves_f1_wt

#---------------------------------------------------------------------------------------------
#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 456 \
#  --batch-size 12 \
#  --traits fruits \
#  --model-dir data/local/effnet_456_fruits_prec_wt

#./phenobase/score_model.py \
#  --trait-csv data/split_all_3.csv \
#  --image-dir data/images/images_600 \
#  --output-csv data/score.csv \
#  --image-size 456 \
#  --batch-size 12 \
#  --traits flowers \
#  --model-dir data/local/effnet_456_flowers_prec
