#!/bin/bash

./phenobase/ensemble_score.py \
  --dataset-csv datasets/splits_2025-04-22.csv \
  --image-dir data/images \
  --checkpoint data/models/ensembles/best_3combo_fract/flowers_2025-04-16/vit_384_lg_flowers_f1_slurm_sl/checkpoint-9450 \
  --output-csv data/inference/vit_384_lg_flowers_f1_slurm_sl.csv \
  --image-size 384 \
  --trait flowers \
  --problem-type single_label_classification

./phenobase/ensemble_score.py \
  --dataset-csv datasets/splits_2025-04-22.csv \
  --image-dir data/images \
  --checkpoint data/models/ensembles/best_3combo_fract/flowers_2025-04-04/vit_384_lg_flowers_f1_a/checkpoint-19199 \
  --output-csv data/inference/vit_384_lg_flowers_f1_a.csv \
  --image-size 384 \
  --trait flowers \
  --problem-type regression

./phenobase/ensemble_score.py \
  --dataset-csv datasets/splits_2025-04-22.csv \
  --image-dir data/images \
  --checkpoint data/models/ensembles/best_3combo_fract/flowers_2025-04-04/effnet_528_flowers_reg_f1_a/checkpoint-17424 \
  --output-csv data/inference/effnet_528_flowers_reg_f1_a.csv \
  --image-size 528 \
  --trait flowers \
  --problem-type regression
