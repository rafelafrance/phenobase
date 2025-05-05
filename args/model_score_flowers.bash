#!/bin/bash

#---------------------------------------------------------------------------------------------
score="data/scores/flowers/effnet_528_flowers_acc_reg.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --trait flowers \
    --model-dir data/models/flowers_2025-04-28/effnet_528_flowers_acc_reg \
    --training-log data/logs/effnet_528_flowers_acc_reg_66369774.out \
    --score-json "$score" \
    --image-size 528 \
    --problem-type regression
fi

score="data/scores/flowers/effnet_528_flowers_acc_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-28/effnet_528_flowers_acc_sl \
    --training-log data/logs/effnet_528_flowers_acc_sl_66262519.out \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_600_flowers_acc_loc_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-28/effnet_600_flowers_acc_loc_sl \
    --score-json "$score" \
    --image-size 600 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_600_flowers_acc_reg.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --trait flowers \
    --model-dir data/models/flowers_2025-04-28/effnet_600_flowers_acc_reg \
    --training-log data/logs/effnet_600_flowers_acc_reg_66369781.out \
    --score-json "$score" \
    --image-size 600 \
    --problem-type regression
fi

score="data/scores/flowers/effnet_600_flowers_acc_sl_66264047.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-28/effnet_600_flowers_acc_sl \
    --training-log data/logs/effnet_600_flowers_acc_sl_66264047.out \
    --score-json "$score" \
    --image-size 600 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/vit_224_lg_flowers_acc_reg.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --trait flowers \
    --model-dir data/models/flowers_2025-04-28/vit_224_lg_flowers_acc_reg \
    --training-log data/logs/vit_224_lg_flowers_acc_reg_66467704.out \
    --score-json "$score" \
    --image-size 224 \
    --problem-type regression
fi

score="data/scores/flowers/vit_224_lg_flowers_acc_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --trait flowers \
    --model-dir data/models/flowers_2025-04-28/vit_224_lg_flowers_acc_sl \
    --training-log data/logs/vit_224_lg_flowers_acc_sl_66467708.out \
    --score-json "$score" \
    --image-size 224 \
    --problem-type single_label_classification
fi

score="data/scores/flowers/vit_384_lg_flowers_acc_reg.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --trait flowers \
    --model-dir data/models/flowers_2025-04-28/vit_384_lg_flowers_acc_reg \
    --training-log data/logs/vit_384_lg_flowers_acc_reg_66369793.out \
    --score-json "$score" \
    --image-size 384 \
    --problem-type regression
fi

score="data/scores/flowers/vit_384_lg_flowers_acc_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-28/vit_384_lg_flowers_acc_sl \
    --training-log data/logs/vit_384_lg_flowers_acc_sl_66262529.out \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_528_flowers_f1_reg.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/effnet_528_flowers_f1_reg \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/effnet_528_flowers_f1_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/effnet_528_flowers_f1_sl \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_528_flowers_reg_loc_f1_hf.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/effnet_528_flowers_reg_loc_f1_hf \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/effnet_600_flowers_f1_loc_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/effnet_600_flowers_f1_loc_sl \
    --score-json "$score" \
    --image-size 600 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/vit_384_base_flowers_f1_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/vit_384_base_flowers_f1_sl \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/vit_384_base_flowers_reg_loc_f1.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/vit_384_base_flowers_reg_loc_f1 \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/vit_384_lg_flowers_f1_loc_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/vit_384_lg_flowers_f1_loc_sl \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/vit_384_lg_flowers_f1_slurm_sl.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-16/vit_384_lg_flowers_f1_slurm_sl \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_528_flowers_reg_f1_a.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-04/effnet_528_flowers_reg_f1_a \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/effnet_528_flowers_sl_f1_a.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-04/effnet_528_flowers_sl_f1_a \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_528_flowers_unk_f1_a.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-04/effnet_528_flowers_unk_f1_a \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/vit_384_lg_flowers_f1_a.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-04/vit_384_lg_flowers_f1_a \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/vit_384_lg_flowers_sl_f1_a.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-04-04/vit_384_lg_flowers_sl_f1_a \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type single_label_classification
fi

#---------------------------------------------------------------------------------------------
score="data/scores/flowers/effnet_600_flowers_f1_loc_sl_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_600_flowers_f1_loc_sl_51 \
    --score-json "$score" \
    --image-size 600 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_600_flowers_f1_reg_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_600_flowers_f1_reg_51 \
    --score-json "$score" \
    --image-size 600 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/effnet_600_flowers_f1_sl_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_600_flowers_f1_sl_51 \
    --score-json "$score" \
    --image-size 600 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/vit_384_lg_flowers_f1_reg_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/vit_384_lg_flowers_f1_reg_51 \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/vit_384_lg_flowers_f1_sl_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/vit_384_lg_flowers_f1_sl_51 \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_528_flowers_f1_loc_unk_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_528_flowers_f1_loc_unk_51 \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/effnet_528_flowers_f1_reg_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_528_flowers_f1_reg_51 \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/effnet_528_flowers_f1_sl_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_528_flowers_f1_sl_51 \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/vit_224_lg_flowers_f1_reg_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/vit_224_lg_flowers_f1_reg_51 \
    --score-json "$score" \
    --image-size 224 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/vit_224_lg_flowers_f1_sl_51.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/vit_224_lg_flowers_f1_sl_51 \
    --score-json "$score" \
    --image-size 224 \
    --trait flowers \
    --problem-type single_label_classification
fi

score="data/scores/flowers/effnet_528_flowers_f1_unk_52.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_528_flowers_f1_unk_52 \
    --score-json "$score" \
    --image-size 528 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/effnet_600_flowers_f1_unk_52.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/effnet_600_flowers_f1_unk_52 \
    --score-json "$score" \
    --image-size 600 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/vit_224_lg_flowers_f1_unk_52.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/vit_224_lg_flowers_f1_unk_52 \
    --score-json "$score" \
    --image-size 224 \
    --trait flowers \
    --problem-type regression
fi

score="data/scores/flowers/vit_384_lg_flowers_f1_unk_52.json"
if [ ! -f "$score" ]; then
./phenobase/model_score.py \
    --dataset-csv datasets/splits_2025-04-22.csv \
    --image-dir data/images \
    --model-dir data/models/flowers_2025-05-02/vit_384_lg_flowers_f1_unk_52 \
    --score-json "$score" \
    --image-size 384 \
    --trait flowers \
    --problem-type regression
fi
