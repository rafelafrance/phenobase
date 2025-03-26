#!/bin/bash

./phenobase/split_data.py \
    --ant-csv datasets/ant/flower_test/fl_abs_eq_ant.csv \
    --ant-csv datasets/ant/flower_test/fl_abs_un_ant.csv \
    --ant-csv datasets/ant/flower_test/fl_pres_eq_ant.csv \
    --ant-csv datasets/ant/flower_test/fl_pres_un_ant.csv \
    --bad-flower-families datasets/bad_families/bad_flower_fams.csv \
    --bad-fruit-families datasets/bad_families/bad_fruit_fams.csv \
    --metadata-db data/gbif_2024-10-28.sqlite \
    --image-dir ../../images/herbarium_sheets \
    --split1 test \
    --gbif-db \
    --split-csv datasets/test_data.csv
