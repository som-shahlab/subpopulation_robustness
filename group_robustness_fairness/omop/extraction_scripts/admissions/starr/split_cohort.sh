#!/bin/bash


DATA_PATH="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130/cohort"

python -m group_robustness_fairness.omop.split_cohort \
    --source_cohort_path=$DATA_PATH'/cohort.parquet' \
    --target_cohort_path=$DATA_PATH'/cohort_fold_1_5.parquet' \
    --test_frac=0.25 \
    --nfold=5 \
    --seed=5573