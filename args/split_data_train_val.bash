#!/bin/bash

./phenobase/split_data.py \
  --ant-csv datasets/ant/ant_0.csv \
  --ant-csv datasets/ant/ant_1.csv \
  --ant-csv datasets/ant/ant_2.csv \
  --ant-csv datasets/ant/ant_3.1.csv \
  --ant-csv datasets/ant/ant_3.2.csv \
  --ant-csv datasets/ant/ant_12.1.24.csv \
  --ant-csv datasets/ant/bud_ant1_12.15.24.csv \
  --bad-flower-families datasets/bad_families/bad_flower_fams.csv \
  --bad-fruit-families datasets/bad_families/bad_fruit_fams.csv \
  --metadata-db data/angiosperms.sqlite \
  --image-dir ../../images/herbarium_sheets \
  --split1 train \
  --split2 val \
  --split-csv datasets/train_data.csv
