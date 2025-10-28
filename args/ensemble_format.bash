#!/bin/bash

./phenobase/ensemble_format.py \
  --winners-csv data/inference/flower_inference_votes_2025-09-02.csv \
  --output-csv data/inference/flower_inference_formatted_2025-10-23.csv \
  --gbif-db data/gbif_2024-10-28.sqlite \
  --model-uri 'doi:10.5281/zenodo.17079402'
