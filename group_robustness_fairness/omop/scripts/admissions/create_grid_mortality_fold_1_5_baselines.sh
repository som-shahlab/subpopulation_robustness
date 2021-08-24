
DATA_PATH="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130"
COHORT_PATH=$DATA_PATH'/cohort/cohort_fold_1_5.parquet'
EXPERIMENT_NAME='mortality_fold_1_5'

python -m group_robustness_fairness.omop.create_grid_baselines \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --experiment_name_prefix=$EXPERIMENT_NAME \
    --tasks 'hospital_mortality' \
    --num_folds=5