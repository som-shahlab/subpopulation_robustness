
DATA_PATH="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130"


EXPERIMENT_NAME_PREFIX='los_fold_1_5'
SOURCE_CONFIG_PATH='/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130/experiments/'$EXPERIMENT_NAME_PREFIX'_selected/'$EXPERIMENT_NAME_PREFIX'_erm_tuning/selected_config.yaml'
TASK='LOS_7'
python -m group_robustness_fairness.omop.create_grid_dro \
    --data_path=$DATA_PATH \
    --experiment_name_prefix=$EXPERIMENT_NAME_PREFIX \
    --source_config_path=$SOURCE_CONFIG_PATH \
    --task $TASK \
    --num_folds=5

EXPERIMENT_NAME_PREFIX='readmission_fold_1_5'
SOURCE_CONFIG_PATH='/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130/experiments/'$EXPERIMENT_NAME_PREFIX'_selected/'$EXPERIMENT_NAME_PREFIX'_erm_tuning/selected_config.yaml'
TASK='readmission_30'
python -m group_robustness_fairness.omop.create_grid_dro \
    --data_path=$DATA_PATH \
    --experiment_name_prefix=$EXPERIMENT_NAME_PREFIX \
    --source_config_path=$SOURCE_CONFIG_PATH \
    --task $TASK \
    --num_folds=5


EXPERIMENT_NAME_PREFIX='mortality_fold_1_5'
SOURCE_CONFIG_PATH='/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130/experiments/'$EXPERIMENT_NAME_PREFIX'_selected/'$EXPERIMENT_NAME_PREFIX'_erm_tuning/selected_config.yaml'
TASK='hospital_mortality'
python -m group_robustness_fairness.omop.create_grid_dro \
    --data_path=$DATA_PATH \
    --experiment_name_prefix=$EXPERIMENT_NAME_PREFIX \
    --source_config_path=$SOURCE_CONFIG_PATH \
    --task $TASK \
    --num_folds=5
