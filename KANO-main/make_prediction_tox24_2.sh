#!/bin/bash

train_path="/KANO-main/data/train_kano_input.csv"
test_path="/KANO-main/data/test_kano_input.csv"
leader_path="/KANO-main/data/leader_kano_input.csv"
checkpoint_dir_base="/KANO-main/dumped/Tox24_2/Tox24_fold_"
preds_path_base="/KANO-main/dumped/Tox24_2/Tox24_fold_"

# Loops over folds and models
for fold in {1..5}; do
    for run in {0..4}; do
        checkpoint_dir="${checkpoint_dir_base}${fold}/run_${run}/model_0/"
        preds_path="${preds_path_base}${fold}/run_${run}/model_0/"
        
        python get_predict_regression.py --gpu 0 \
            --test_path "${train_path}" \
            --checkpoint_dir "${checkpoint_dir}" \
            --preds_path "${preds_path}"

        python get_predict_regression.py --gpu 0 \
            --test_path "${test_path}" \
            --checkpoint_dir "${checkpoint_dir}" \
            --preds_path "${preds_path}"

        python get_predict_regression.py --gpu 0 \
            --test_path "${leader_path}" \
            --checkpoint_dir "${checkpoint_dir}" \
            --preds_path "${preds_path}"
    done
done

