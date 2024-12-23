#!/bin/bash

train_path="/KANO-main/data/train_kano_input.csv"
test_path="/KANO-main/data/test_kano_input.csv"
leader_path="/KANO-main/data/leader_kano_input.csv"
checkpoint_dir_base="/KANO-main/dumped/Tox24_1/"
preds_path_base="/KANO-main/dumped/Tox24_1/"

# Loops over folds and models
for fold in {0..4}; do
    for model in {1..5}; do
        checkpoint_dir="${checkpoint_dir_base}/model_${fold}_${model}/"
        preds_path="${preds_path_base}/model_${fold}_${model}/"
        
        python get_predict_regression.py --gpu 0 \
            --test_path "${train_path}" \
            --checkpoint_dir "${checkpoint_dir}" \
            --preds_path "${preds_path}"

	python get_predict_regression.py --gpu 0 \
            --test_path "${test_path}" \
            --checkpoint_dir "${checkpoint_dir}" \
            --preds_path "${preds_path}"

	python get_predict_regression.py --gpu 0 \
            --test_path "${test_path}" \
            --checkpoint_dir "${checkpoint_dir}" \
            --preds_path "${preds_path}"
    done
done

