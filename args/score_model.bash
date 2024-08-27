#!/bin/bash

####################################################################################################
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/tuned/mae_flowers/checkpoint-21

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/tuned/mae_flowers/checkpoint-50379

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/tuned/mae_flowers/checkpoint-50400

#----------------------------------------------------------------------------------
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/tuned/no_mae_flowers/checkpoint-21

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/tuned/no_mae_flowers/checkpoint-50379

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/tuned/no_mae_flowers/checkpoint-50400
