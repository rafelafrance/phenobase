#!/bin/bash

./phenobase/append_old_data.py \
  --ant-csv datasets/ant/ant_0.csv \
  --ant-csv datasets/ant/ant_1.csv \
  --ant-csv datasets/ant/ant_2.csv \
  --ant-csv datasets/ant/ant_3.1.csv \
  --ant-csv datasets/ant/ant_3.2.csv \
  --ant-csv datasets/ant/ant_12.1.24.csv \
  --ant-csv datasets/ant/bud_ant1_12.15.24.csv \
  --in-csv datasets/all_traits.csv \
  --out-csv datasets/all_traits_v2.csv
