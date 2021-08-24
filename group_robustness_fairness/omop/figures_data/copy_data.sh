#!/bin/bash


DATA_PATH='/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/'

cp \
    $DATA_PATH'admissions/starr_20210130/experiments/los_fold_1_5_selected/result_df_ci_no_agg.csv' \
    './admissions/los/'

cp \
    $DATA_PATH'admissions/starr_20210130/experiments/mortality_fold_1_5_selected/result_df_ci_no_agg.csv' \
    './admissions/mortality/'

cp \
    $DATA_PATH'admissions/starr_20210130/experiments/readmission_fold_1_5_selected/result_df_ci_no_agg.csv' \
    './admissions/readmission/'
