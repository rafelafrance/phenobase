#!/bin/bash

####################################################################################################
./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_all.csv \
  --trait flowers \
  --trait fruits \
  --trait leaves \
  --trait whole_plant \
  --trait reproductive_structure \
  --pretrained-dir data/training_output/vit_all/checkpoint-663

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_all.csv \
  --trait flowers \
  --trait fruits \
  --trait leaves \
  --trait whole_plant \
  --trait reproductive_structure \
  --pretrained-dir data/training_output/vit_all/checkpoint-15587

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_all.csv \
  --trait flowers \
  --trait fruits \
  --trait leaves \
  --trait whole_plant \
  --trait reproductive_structure \
  --pretrained-dir data/training_output/vit_all/checkpoint-15600

####################################################################################################
./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers.csv \
  --trait flowers \
  --pretrained-dir data/training_output/vit_flowers/checkpoint-494

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers.csv \
  --trait flowers \
  --pretrained-dir data/training_output/vit_flowers/checkpoint-15587

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers.csv \
  --trait flowers \
  --pretrained-dir data/training_output/vit_flowers/checkpoint-15600

####################################################################################################
./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers_fruits.csv \
  --trait flowers \
  --trait fruits \
  --pretrained-dir data/training_output/vit_flowers_fruits/checkpoint-403

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers_fruits.csv \
  --trait flowers \
  --trait fruits \
  --pretrained-dir data/training_output/vit_flowers_fruits/checkpoint-15587

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers_fruits.csv \
  --trait flowers \
  --trait fruits \
  --pretrained-dir data/training_output/vit_flowers_fruits/checkpoint-15600

####################################################################################################
./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers_no_mae.csv \
  --trait flowers \
  --pretrained-dir data/training_output/vit_flowers_no_mae/checkpoint-117

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers_no_mae.csv \
  --trait flowers \
  --pretrained-dir data/training_output/vit_flowers_no_mae/checkpoint-15587

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_flowers_no_mae.csv \
  --trait flowers \
  --pretrained-dir data/training_output/vit_flowers_no_mae/checkpoint-15600

####################################################################################################
./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_fruits.csv \
  --trait fruits \
  --pretrained-dir data/training_output/vit_fruits/checkpoint-1092

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_fruits.csv \
  --trait fruits \
  --pretrained-dir data/training_output/vit_fruits/checkpoint-14729

./phenobase/score_model.py \
  --trait-csv data/split_all.csv \
  --image-dir data/images/images_224 \
  --output-csv data/vit_fruits.csv \
  --trait fruits \
  --pretrained-dir data/training_output/vit_fruits/checkpoint-14742
