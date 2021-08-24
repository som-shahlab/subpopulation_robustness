# group_robustness_fairness.omop

## Overview of prediction tasks and datasets
    * Inpatient cohort (In-hospital mortality, prolonged length of stay (7 days), 30-day readmission)
        * Stanford Medicine Research Repository (STARR)

## Sub-package structure
    * `group_robustness_fairness.omop`
        * Base directory with `.py` files
    * `group_robustness_fairness.omop.extraction_scripts`
        * Shell scripts that extract cohorts and features
    * `group_robustness_fairness.omop.notebooks`
        * Jupyter notebooks
    * `group_robustness_fairness.omop.scripts`
        * Shell scripts that execute all experiments downstream of data extraction


## Execution workflow
    * Inpatient Cohort
        * Extract data
            * `extraction_scripts/admissions/create_cohort_starr.sh`
            * `extraction_scripts/admissions/extract_features.sh`
            * `extraction_scripts/admissions/split_cohort.sh`
        * Training and evaluation pipeline
            * `scripts/admissions/create_grid_los_fold_1_5_baselines.sh`
            * `scripts/admissions/create_grid_mortality_fold_1_5_baselines.sh`
            * `scripts/admissions/create_grid_readmission_fold_1_5_baselines.sh`
            * `scripts/admissions/run_tuning.sh`
            * `scripts/admissions/get_best_model_erm.sh`
            * `scripts/admissions/create_grids_dro.sh`
            * `scripts/admissions/run_tuning_dro.sh`
            * `scripts/admissions/get_best_model_all.sh`
            * `scripts/admissions/bootstrapping_replicates.sh`
    * Make figures and tables (for all results, including MIMIC/eICU)
        * notebooks/make_plots.ipynb (requires jupyter notebook with R kernel)
        * notebooks/create_cohort_tables.ipynb
        * notebooks/create_cohort_tables_mimic_eicu.ipynb

