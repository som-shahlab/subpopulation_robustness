#!/bin/bash

train_model_func() {
    python -m group_robustness_fairness.omop.run_tuning \
        --experiment_name=$1'_fold_1_5_erm_tuning' \
        --cuda_device=$2 \
        --max_concurrent_jobs=5 \
        --num_workers=0 \
        --cohort_name="cohort_fold_1_5.parquet"

    python -m group_robustness_fairness.omop.run_tuning \
        --experiment_name=$1'_fold_1_5_erm_group_aware' \
        --cuda_device=$2 \
        --max_concurrent_jobs=5 \
        --num_workers=0 \
        --cohort_name="cohort_fold_1_5.parquet"

    python -m group_robustness_fairness.omop.run_tuning \
        --experiment_name=$1'_fold_1_5_erm_subset_tuning' \
        --cuda_device=$2 \
        --max_concurrent_jobs=5 \
        --num_workers=0 \
        --cohort_name="cohort_fold_1_5.parquet"
}


train_model_func 'los' 5 &
train_model_func 'readmission' 6 &
train_model_func 'mortality' 7