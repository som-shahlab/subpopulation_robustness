#!/bin/bash

DATASET="starr_omop_cdm5_deid_20210130"
RS_DATASET="plp_cohort_tables"

GCLOUD_PROJECT="som-nero-nigam-starr"
DATASET_PROJECT="som-nero-nigam-starr"
RS_DATASET_PROJECT="som-nero-nigam-starr"

COHORT_NAME="admission_rollup_filtered_"$DATASET
DATA_PATH="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130"

FEATURES_DATASET="temp_dataset"
FEATURES_PREFIX="features_"$USER
INDEX_DATE_FIELD='admit_date'
ROW_ID_FIELD='prediction_id'
MERGED_NAME='merged_features_binary'

python -m group_robustness_fairness.prediction_utils.extraction_utils.extract_features \
    --data_path=$DATA_PATH \
    --features_by_analysis_path="features_by_analysis" \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --features_dataset=$FEATURES_DATASET \
    --features_prefix=$FEATURES_PREFIX \
    --index_date_field=$INDEX_DATE_FIELD \
    --row_id_field=$ROW_ID_FIELD \
    --merged_name=$MERGED_NAME \
    --analysis_ids \
        "condition_occurrence_delayed" \
        "procedure_occurrence_delayed" \
        "drug_exposure" \
        "device_exposure" \
        "measurement" \
        "note_type_delayed" \
        "observation" \
        "measurement_range" \
        "gender" \
        "race" \
        "ethnicity" \
        "age_group" \
    --time_bins -365 -30 0 \
    --binary \
    --featurize \
    --no_cloud_storage \
    --merge_features \
    --create_sparse \
    --no_create_parquet \
    --overwrite