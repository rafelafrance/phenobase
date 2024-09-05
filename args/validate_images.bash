#!/bin/bash

python ./phenobase/validate_images.py \
  --sheets-dir ../../images/herbarium_sheets \
  --cull-dir ../../images/cull \
  --resize-dir ./data/images/images_456 \
  --resize 456 \
  --cull-bad-images
