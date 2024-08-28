#!/bin/bash

####################################################################################################
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/mae_flowers/checkpoint-3087
#
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/mae_flowers/checkpoint-15183
#
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/mae_flowers/checkpoint-15204

#----------------------------------------------------------------------------------
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/no_mae_flowers/checkpoint-18606
#
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/no_mae_flowers/checkpoint-50379
#
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/no_mae_flowers/checkpoint-50400

#----------------------------------------------------------------------------------
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/backups/training_output/vit_flowers_no_mae/checkpoint-117

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/backups/training_output/vit_flowers_no_mae/checkpoint-15587

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/backups/training_output/vit_flowers_no_mae/checkpoint-15600
