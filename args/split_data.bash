#!/bin/bash

./phenobase/split_data.py \
  --ant-csv splits/ant/ant_0.csv \
  --ant-csv splits/ant/ant_1.csv \
  --ant-csv splits/ant/ant_2.csv \
  --ant-csv splits/ant/ant_3.1.csv \
  --ant-csv splits/ant/ant_3.2.csv \
  --ant-csv splits/ant/ant_12.1.24.csv \
  --ant-csv splits/ant/bud_ant1_12.15.24.csv \
  --bad-flower-families splits/families/bad_flower_fams.csv \
  --bad-fruit-families splits/families/bad_fruit_fams.csv \
  --metadata-db data/angiosperms.sqlite \
  --split-dir splits
