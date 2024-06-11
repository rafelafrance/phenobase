#!/bin/bash

script="../transformers/examples/pytorch/image-pretraining/run_mae.py"

python $script \
  --train_dir data/mae_splits_224/train \
  --validation_dir data/mae_splits_224/validation \
  --output_dir data/output/local \
  --dataset_name data/mae_splits_224 \
  --do_train \
  --do_eval
