import pandas as pd
import glob
import os
import configargparse as argparse
from group_robustness_fairness.prediction_utils.util import df_dict_concat, yaml_write


parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--project_dir",
    type=str,
    default="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130",
)

parser.add_argument("--task_prefix", type=str, required=True)

parser.add_argument(
    "--selected_config_experiment_suffix", type=str, default="selected",
)

if __name__ == "__main__":

    args = parser.parse_args()
    project_dir = args.project_dir
    task_prefix = args.task_prefix

    def get_config_df(experiment_name):
        config_df_path = os.path.join(
            os.path.join(
                project_dir, "experiments", experiment_name, "config", "config.csv"
            )
        )
        config_df = pd.read_csv(config_df_path)
        config_df["config_filename"] = config_df.id.astype(str) + ".yaml"
        return config_df

    def get_result_df(
        experiment_name, output_filename="result_df_group_standard_eval.parquet"
    ):
        baseline_files = glob.glob(
            os.path.join(
                project_dir, "experiments", experiment_name, "**", output_filename
            ),
            recursive=True,
        )
        assert len(baseline_files) > 0
        baseline_df_dict = {
            tuple(file_name.split("/"))[-2]: pd.read_parquet(file_name)
            for file_name in baseline_files
        }
        baseline_df = df_dict_concat(baseline_df_dict, ["config_filename"])
        return baseline_df

    def append_hparams_id_col(config_df):
        config_df = (
            config_df[
                set(config_df.columns) - set(["id", "config_filename", "fold_id"])
            ]
            .drop_duplicates(ignore_index=True)
            .rename_axis("hparams_id")
            .reset_index()
            .merge(config_df)
        )
        return config_df

    experiment_names = {
        "erm_baseline": "{}_erm_tuning".format(task_prefix),
    }

    config_df_erm = get_config_df(experiment_names["erm_baseline"])
    result_df_erm = get_result_df(
        experiment_names["erm_baseline"],
        output_filename="result_df_training_eval.parquet",
    )
    config_df_erm = append_hparams_id_col(config_df_erm)
    result_df_erm = result_df_erm.merge(
        config_df_erm[["config_filename", "hparams_id"]]
    )

    mean_performance = pd.DataFrame(
        result_df_erm.merge(config_df_erm)
        .query('metric == "loss_bce" & phase == "eval"')
        .groupby(["hparams_id"])
        .agg(performance=("performance", "mean"))
        .reset_index()
    )

    best_model = mean_performance.agg(performance=("performance", "min")).merge(
        mean_performance
    )

    selected_config_df = best_model[["hparams_id"]].merge(config_df_erm)
    selected_config_dict_list = (
        selected_config_df.drop(
            columns=["id", "fold_id", "config_filename", "hparams_id"]
        )
        .drop_duplicates()
        .to_dict("records")
    )
    assert len(selected_config_dict_list) == 1
    selected_config_dict = selected_config_dict_list[0]
    print(selected_config_dict)

    selected_config_experiment_name = "{}_{}".format(
        task_prefix, args.selected_config_experiment_suffix
    )
    selected_config_experiment_path = os.path.join(
        project_dir,
        "experiments",
        selected_config_experiment_name,
        experiment_names["erm_baseline"],
    )

    os.makedirs(selected_config_experiment_path, exist_ok=True)
    selected_config_df.to_csv(
        os.path.join(selected_config_experiment_path, "selected_configs.csv"),
        index=False,
    )
    yaml_write(
        selected_config_dict,
        os.path.join(selected_config_experiment_path, "selected_config.yaml"),
    )
