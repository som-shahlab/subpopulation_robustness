#!/bin/bash

OUTPUT_PATH=/home/spfohl/slurm_out/create_cohort_starr.out

DATASET="starr_omop_cdm5_deid_20210130"
RS_DATASET="plp_cohort_tables"

GCLOUD_PROJECT="som-nero-nigam-starr"
DATASET_PROJECT="som-nero-nigam-starr"
RS_DATASET_PROJECT="som-nero-nigam-starr"

COHORT_NAME="admission_rollup_"$DATASET
COHORT_NAME_LABELED="admission_rollup_labeled_"$DATASET
COHORT_NAME_FILTERED="admission_rollup_filtered_"$DATASET

DATA_PATH="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130"

python -m group_robustness_fairness.prediction_utils.cohorts.admissions.create_cohort \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --cohort_name_labeled=$COHORT_NAME_LABELED \
    --cohort_name_filtered=$COHORT_NAME_FILTERED \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --data_path=$DATA_PATH \
    --has_birth_datetime