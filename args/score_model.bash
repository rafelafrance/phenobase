#!/bin/bash

####################################################################################################
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/no_mae/flowers/checkpoint-84

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/no_mae/flowers/checkpoint-50379

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/no_mae/flowers/checkpoint-50400

#----------------------------------------------------------------------------------
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/training_output/vit_flowers/checkpoint-494

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/training_output/vit_flowers/checkpoint-15587

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/training_output/vit_flowers/checkpoint-15600

#----------------------------------------------------------------------------------
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/training_output/vit_flowers_no_mae/checkpoint-117

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/training_output/vit_flowers_no_mae/checkpoint-15587

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait flowers \
 --pretrained-dir data/training_output/vit_flowers_no_mae/checkpoint-15600

####################################################################################################
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait fruits \
 --pretrained-dir data/no_mae/fruits/checkpoint-76

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait fruits \
 --pretrained-dir data/no_mae/fruits/checkpoint-45581

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait fruits \
 --pretrained-dir data/no_mae/fruits/checkpoint-45600

#----------------------------------------------------------------------------------
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait fruits \
 --pretrained-dir data/training_output/vit_fruits/checkpoint-1092

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait fruits \
 --pretrained-dir data/training_output/vit_fruits/checkpoint-14729

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait fruits \
 --pretrained-dir data/training_output/vit_fruits/checkpoint-14742

####################################################################################################
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait leaves \
 --pretrained-dir data/no_mae/leaves/checkpoint-72

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait leaves \
 --pretrained-dir data/no_mae/leaves/checkpoint-57576

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait leaves \
 --pretrained-dir data/no_mae/leaves/checkpoint-57600

####################################################################################################
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait reproductive_structure \
 --pretrained-dir data/no_mae/reproductive_structure/checkpoint-184

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait reproductive_structure \
 --pretrained-dir data/no_mae/reproductive_structure/checkpoint-55177

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait reproductive_structure \
 --pretrained-dir data/no_mae/reproductive_structure/checkpoint-55200

####################################################################################################
./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait whole_plant \
 --pretrained-dir data/no_mae/whole_plant/checkpoint-120

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait whole_plant \
 --pretrained-dir data/no_mae/whole_plant/checkpoint-57576

./phenobase/score_model.py \
 --trait-csv data/split_all_3.csv \
 --image-dir data/images/images_224 \
 --output-csv data/score.csv \
 --trait whole_plant \
 --pretrained-dir data/no_mae/whole_plant/checkpoint-57600
