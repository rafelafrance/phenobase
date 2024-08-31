#!/bin/bash

python ./phenobase/validate_images.py \
  --sheets-dir ../../images/herbarium_sheets \
  --cull-dir ../../images/cull \
  --resize-dir ./data/images/images_600 \
  --resize 600 \
  --cull-bad-images
