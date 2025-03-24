#!/bin/bash

#SBATCH --account=guralnick
#SBATCH --qos=guralnick

#SBATCH --job-name=pretraining_mae

#SBATCH --mail-user=rafe.lafrance@ufl.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/blue/guralnick/rafe.lafrance/phenobase/logs/%x_%j.out

#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

date;hostname;pwd

export PATH=/blue/guralnick/rafe.lafrance/.conda/envs/vitmae/bin:$PATH

module purge

python3 /blue/guralnick/rafe.lafrance/phenobase/phenobase/run_mae.py \
    --dataset_name /blue/guralnick/rafe.lafrance/phenobase/data/images/images_600 \
    --output_dir /blue/guralnick/rafe.lafrance/phenobase/data/pretraining_output \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 800 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 9702236

date
