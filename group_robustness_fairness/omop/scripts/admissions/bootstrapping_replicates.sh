#!/bin/bash 

DATA_PATH="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130"
COHORT_PATH=$DATA_PATH"/cohort/cohort_fold_1_5.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"
N_BOOT=1000
SEED=718

TASK="hospital_mortality"
SELECTED_CONFIG_EXPERIMENT_NAME="mortality_fold_1_5_selected"
BASELINE_EXPERIMENT_NAME="mortality_fold_1_5_erm_tuning"

python -m group_robustness_fairness.omop.bootstrapping_replicates \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --n_boot=$N_BOOT \
    --seed=$SEED \
    --task=$TASK \
    --selected_config_experiment_name=$SELECTED_CONFIG_EXPERIMENT_NAME \
    --baseline_experiment_name=$BASELINE_EXPERIMENT_NAME \
    --n_jobs=8

TASK="readmission_30"
SELECTED_CONFIG_EXPERIMENT_NAME="readmission_fold_1_5_selected"
BASELINE_EXPERIMENT_NAME="readmission_fold_1_5_erm_tuning"

python -m group_robustness_fairness.omop.bootstrapping_replicates \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --n_boot=$N_BOOT \
    --seed=$SEED \
    --task=$TASK \
    --selected_config_experiment_name=$SELECTED_CONFIG_EXPERIMENT_NAME \
    --baseline_experiment_name=$BASELINE_EXPERIMENT_NAME \
    --n_jobs=8

TASK="LOS_7"
SELECTED_CONFIG_EXPERIMENT_NAME="los_fold_1_5_selected"
BASELINE_EXPERIMENT_NAME="los_fold_1_5_erm_tuning"

python -m group_robustness_fairness.omop.bootstrapping_replicates \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --n_boot=$N_BOOT \
    --seed=$SEED \
    --task=$TASK \
    --selected_config_experiment_name=$SELECTED_CONFIG_EXPERIMENT_NAME \
    --baseline_experiment_name=$BASELINE_EXPERIMENT_NAME \
    --n_jobs=8