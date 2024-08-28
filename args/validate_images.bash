#!/bin/bash

python ./phenobase/validate_images.py \
  --sheets-dir ../../images/herbarium_sheets \
  --cull-dir ../../images/cull \
  --resize-dir ./data/images/images_384 \
  --resize 384 \
  --cull-bad-images
