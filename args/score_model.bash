#!/bin/bash

####################################################################################################
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/mae_flowers

#----------------------------------------------------------------------------------
#./phenobase/score_model.py \
# --trait-csv data/split_all_3.csv \
# --image-dir data/images/images_224 \
# --output-csv data/score.csv \
# --trait flowers \
# --pretrained-dir data/tuned/no_mae_flowers

#----------------------------------------------------------------------------------
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/backups/training_output
