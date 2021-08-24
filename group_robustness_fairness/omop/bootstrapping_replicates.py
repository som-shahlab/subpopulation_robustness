import numpy as np
import pandas as pd
import os
import argparse
import random
from group_robustness.fairness.prediction_utils.pytorch_utils.metrics import (
    StandardEvaluator,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    type=str,
    default="",
    help="The root path where data is stored",
    required=True,
)
parser.add_argument(
    "--cohort_path",
    type=str,
    default="",
    help="File name for the file containing label information",
    required=True,
)
parser.add_argument("--features_row_id_map_path", type=str, default="", required=True)

parser.add_argument(
    "--eval_attributes",
    type=str,
    nargs="+",
    required=False,
    default=["race_eth", "gender_concept_name", "age_group"],
    help="The attributes to use to perform a stratified evaluation. Refers to columns in the cohort dataframe",
)
parser.add_argument(
    "--eval_phases", type=str, nargs="+", required=False, default=["test"],
)
parser.add_argument("--task", type=str, required=True)

parser.add_argument(
    "--metrics",
    type=str,
    nargs="+",
    required=False,
    default=["auc", "loss_bce", "ace_abs_logistic_log"],
    help="The attributes to use to perform a stratified evaluation. Refers to columns in the cohort dataframe",
)
parser.add_argument("--n_boot", type=int, default=1000)
parser.add_argument("--seed", type=int, default=718)

parser.add_argument("--selected_config_experiment_name", required=True)
parser.add_argument("--baseline_experiment_name", required=True)

parser.add_argument(
    "--weight_var_name",
    type=str,
    default=None,
    help="The name of a column with weights",
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=None,
    help="The number of cores to use for bootstrapping",
)


def sample_cohort(df, group_vars=["fold_id"]):
    return df.groupby(group_vars).sample(frac=1.0, replace=True).reset_index(drop=True)


def get_output_df(data_path, best_model_config_df, eval_phases=["eval", "test"]):
    output_df_dict = {}
    for i, row in best_model_config_df.iterrows():
        experiment_key = (
            row.model_row_id,
            row.experiment_name,
            row.tag,
            row.hparams_id,
            row.config_filename,
            row.training_fold_id,
        )
        output_df_dict[experiment_key] = pd.read_parquet(
            os.path.join(
                data_path,
                "experiments",
                row.experiment_name,
                "performance",
                row.config_filename,
                "output_df.parquet",
            )
        ).query("phase in @eval_phases")
    return (
        pd.concat(output_df_dict)
        .reset_index(level=-1, drop=True)
        .rename_axis(
            [
                "model_row_id",
                "experiment_name",
                "tag",
                "hparams_id",
                "config_filename",
                "training_fold_id",
            ]
        )
        .reset_index()
    )


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cohort = pd.read_parquet(args.cohort_path)
    features_row_id_map = pd.read_parquet(args.features_row_id_map_path)
    cohort = cohort.merge(features_row_id_map)
    cohort = cohort.rename(columns={"features_row_id": "row_id"})

    eval_phases = args.eval_phases
    cohort_small_cols = (
        ["person_id", "row_id", "fold_id"] + [args.task] + args.eval_attributes
    )
    if args.weight_var_name is not None:
        cohort_small_cols.append(args.weight_var_name)

    cohort_small = cohort[cohort_small_cols].query("fold_id in @eval_phases")

    best_model_config_df_path = os.path.join(
        args.data_path,
        "experiments",
        args.selected_config_experiment_name,
        "selected_configs.csv",
    )

    best_model_config_df = pd.read_csv(best_model_config_df_path)
    best_model_config_df = (
        best_model_config_df.reset_index(drop=True)
        .rename_axis("model_row_id")
        .reset_index()
        .rename(columns={"fold_id": "training_fold_id"})
        .drop(columns="eval_attribute")
    )

    cohort_long = cohort_small.melt(
        id_vars=set(cohort_small.columns) - set(args.eval_attributes),
        value_vars=set(args.eval_attributes),
        var_name="eval_attribute",
        value_name="eval_group",
    )
    output_df = get_output_df(
        args.data_path, best_model_config_df, eval_phases=eval_phases
    )
    output_df = output_df.merge(
        best_model_config_df[
            ["model_row_id", "sensitive_attribute", "subset_attribute", "subset_group"]
        ].drop_duplicates()
    )

    output_df = output_df.merge(
        cohort_long[["row_id", "fold_id", "eval_attribute", "eval_group"]]
    )
    # Apply filters
    # Filter out cases where sensitive attribute != eval_attribute, if sensitive_attribute is defined
    # Filters out subset cases where subset_attribute != eval_attribute or subset_group != eval_group, when defined
    output_df = output_df.query(
        "~(~sensitive_attribute.isnull() & sensitive_attribute != eval_attribute)"
    ).query(
        """~(~subset_attribute.isnull() & ~subset_group.isnull() & (subset_attribute != eval_attribute | subset_group != eval_group))"""
    )

    # Run the bootstrap
    evaluator = StandardEvaluator(metrics=args.metrics)
    result_df_ci = evaluator.bootstrap_evaluate(
        df=output_df,
        n_boot=args.n_boot,
        strata_vars_eval=["tag", "training_fold_id", "eval_attribute", "eval_group"],
        strata_vars_boot=["phase", "labels", "eval_attribute", "eval_group"],
        strata_var_replicate="training_fold_id",
        replicate_aggregation_mode=None,
        strata_var_experiment="tag",
        strata_var_group="eval_group",
        baseline_experiment_name="erm_baseline",
        compute_overall=True,
        compute_group_min_max=True,
        weight_var="weights" if args.weight_var_name is not None else None,
        patient_id_var="row_id",
        n_jobs=args.n_jobs,
        verbose=True,
    )

    result_df_ci_path = os.path.join(
        args.data_path, "experiments", args.selected_config_experiment_name,
    )

    os.makedirs(result_df_ci_path, exist_ok=True)

    result_df_ci.to_csv(
        os.path.join(result_df_ci_path, "result_df_ci_no_agg.csv"), index=False
    )
