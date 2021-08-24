
PROJECT_DIR="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130"
SELECTED_CONFIG_EXPERIMENT_SUFFIX="selected"

TASK_PREFIX="los_fold_1_5"
python -m group_robustness_fairness.omop.get_best_model_erm \
    --project_dir=$PROJECT_DIR \
    --task_prefix=$TASK_PREFIX \
    --selected_config_experiment_suffix=$SELECTED_CONFIG_EXPERIMENT_SUFFIX

TASK_PREFIX="mortality_fold_1_5"
python -m group_robustness_fairness.omop.get_best_model_erm \
    --project_dir=$PROJECT_DIR \
    --task_prefix=$TASK_PREFIX \
    --selected_config_experiment_suffix=$SELECTED_CONFIG_EXPERIMENT_SUFFIX

TASK_PREFIX="readmission_fold_1_5"
python -m group_robustness_fairness.omop.get_best_model_erm \
    --project_dir=$PROJECT_DIR \
    --task_prefix=$TASK_PREFIX \
    --selected_config_experiment_suffix=$SELECTED_CONFIG_EXPERIMENT_SUFFIX