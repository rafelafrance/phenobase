#!/bin/bash

./phenobase/ensemble_archive.py \
  --winners-csv data/inference/flower_inference_votes_2025-09-02.csv \
  --output-csv data/inference/flower_inference_archive_2025-11-20.csv \
  --gbif-db data/gbif_2024-10-28.sqlite \
  --model-uri 'doi:10.5281/zenodo.17079402'
