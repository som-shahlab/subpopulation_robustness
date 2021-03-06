import configargparse as argparse
import os
import copy
from group_robustness_fairness.omop.util import run_commands

parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--data_path",
    type=str,
    default="/local-scratch/nigam/projects/spfohl/group_robustness_fairness/cohorts/admissions/starr_20210130",
)

parser.add_argument(
    "--experiment_name", type=str, required=True,
)

parser.add_argument(
    "--max_concurrent_jobs", type=int, default=4,
)

parser.add_argument("--cuda_device", type=int, default=6)

parser.add_argument("--num_workers", type=int, default=0)

parser.add_argument("--cohort_name", type=str, default="cohort_fold_1_5.parquet")

parser.add_argument("--weight_var_name", type=str, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.data_path
    features_path = os.path.join(
        data_path, "merged_features_binary", "features_sparse", "features.gz"
    )
    cohort_path = os.path.join(data_path, "cohort", args.cohort_name)
    vocab_path = os.path.join(
        data_path, "merged_features_binary", "vocab", "vocab.parquet"
    )
    features_row_id_map_path = os.path.join(
        data_path,
        "merged_features_binary",
        "features_sparse",
        "features_row_id_map.parquet",
    )
    config_dir = os.path.join(data_path, "experiments", args.experiment_name, "config")
    config_filenames = [
        x for x in os.listdir(config_dir) if os.path.splitext(x)[-1] == ".yaml"
    ]

    base_commands = [
        "/local-scratch/nigam/envs/prediction_utils/bin/python",
        "-m",
        "group_robustness_fairness.train_model",
        "--data_path",
        data_path,
        "--features_path",
        features_path,
        "--cohort_path",
        cohort_path,
        "--vocab_path",
        vocab_path,
        "--features_row_id_map_path",
        features_row_id_map_path,
        "--run_evaluation",
        "--run_evaluation_group",
        "--run_evaluation_group_standard",
        "--run_evaluation_group_fair_ova",
        "--eval_attributes",
        "age_group",
        "gender_concept_name",
        "race_eth",
        "--num_workers",
        str(args.num_workers),
        "--cuda_device",
        str(args.cuda_device),
        "--save_outputs",
        "--sparse_mode",
        "list",
    ]

    if args.weight_var_name is not None:
        base_commands = base_commands + ["--weight_var_name", args.weight_var_name]

    commands_list = [
        copy.copy(base_commands)
        + [
            "--config_path",
            os.path.join(config_dir, config_filename),
            "--result_path",
            os.path.join(
                data_path,
                "experiments",
                args.experiment_name,
                "performance",
                config_filename,
            ),
            "--logging_path",
            os.path.join(
                data_path,
                "experiments",
                args.experiment_name,
                "performance",
                config_filename,
                "training_log.log",
            ),
        ]
        for config_filename in config_filenames
    ]
    print(len(commands_list))
    run_commands(commands_list, max_concurrent_jobs=args.max_concurrent_jobs)
