#!/bin/bash

./phenobase/split_data.py \
    --ant-gbif datasets/ant_gbif \
    --ant-idigbio datasets/ant_idigbio \
    --gbif-db data/gbif_2024-10-28.sqlite \
    --idigbio-db data/angiosperms.sqlite \
    --image-dir datasets/images \
    --filter-dir datasets/filter \
    --split-csv datasets/splits_2025-04-22.csv
