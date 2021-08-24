import numpy as np
import os
import random
import pandas as pd
import configargparse as argparse
import itertools

from group_robustness_fairness.prediction_utils.util import yaml_write, yaml_read
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    config_file_parser_class=argparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--data_path", type=str, default="", help="The root data path",
)

parser.add_argument(
    "--experiment_name_prefix",
    type=str,
    default="scratch",
    help="The name of the experiment",
)

parser.add_argument("--source_config_path", type=str, required=True)

parser.add_argument(
    "--grid_size",
    type=int,
    default=None,
    help="The number of elements in the random grid",
)

parser.add_argument(
    "--attributes",
    type=str,
    nargs="*",
    required=False,
    default=["race_eth", "gender_concept_name", "age_group"],
)

parser.add_argument(
    "--task", type=str, required=True,
)

parser.add_argument(
    "--num_folds", type=int, required=False, default=5,
)

parser.add_argument("--seed", type=int, default=234)


def generate_grid(
    global_tuning_params_dict,
    model_tuning_params_dict=None,
    experiment_params_dict=None,
    grid_size=None,
    seed=None,
):

    the_grid = list(ParameterGrid(global_tuning_params_dict))
    if model_tuning_params_dict is not None:
        local_grid = []
        for i, pair_of_grids in enumerate(
            itertools.product(the_grid, list(ParameterGrid(model_tuning_params_dict)))
        ):
            local_grid.append({**pair_of_grids[0], **pair_of_grids[1]})
        the_grid = local_grid

    if grid_size is not None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        np.random.shuffle(the_grid)
        the_grid = the_grid[:grid_size]

    if experiment_params_dict is not None:
        outer_grid = list(ParameterGrid(experiment_params_dict))
        final_grid = []
        for i, pair_of_grids in enumerate(itertools.product(the_grid, outer_grid)):
            final_grid.append({**pair_of_grids[0], **pair_of_grids[1]})
        return final_grid
    else:
        return the_grid


if __name__ == "__main__":

    args = parser.parse_args()
    source_config = yaml_read(args.source_config_path)
    source_config = {
        key: [value] if not isinstance(value, list) else value
        for key, value in source_config.items()
    }

    common_grid = {
        "global": source_config,
        "model_specific": {
            "size_adjusted": {
                "group_objective_metric": [
                    "size_adjusted_loss",
                    "size_adjusted_loss_reciprocal",
                ],
                "adjustment_scale": [0.01, 0.1, 1],
            },
            "not_size_adjusted": {
                "group_objective_metric": ["loss", "baselined_loss", "auc_proxy"]
            },
        },
        "experiment": {
            "label_col": [args.task],
            "fold_id": [str(i + 1) for i in range(args.num_folds)],
        },
    }

    grids = {
        "dro_tuning": {
            "global": {**common_grid["global"], **{"lr_lambda": [1, 1e-1, 1e-2]}},
            "model_specific": [
                common_grid["model_specific"]["size_adjusted"],
                common_grid["model_specific"]["not_size_adjusted"],
            ],
            "experiment": {
                **common_grid["experiment"],
                **{
                    "group_objective_type": ["dro"],
                    "sensitive_attribute": args.attributes,
                    "balance_groups": [True, False],
                    "selection_metric": ["loss", "auc_min", "loss_bce_max"],
                },
            },
        },
    }

    for experiment_name, value in grids.items():
        print(experiment_name)
        the_grid = generate_grid(
            grids[experiment_name]["global"],
            grids[experiment_name]["model_specific"],
            grids[experiment_name]["experiment"],
        )
        print("{}, length: {}".format(experiment_name, len(the_grid)))

        experiment_dir_name = "{}_{}".format(
            args.experiment_name_prefix, experiment_name
        )

        grid_df = pd.DataFrame(the_grid)
        config_path = os.path.join(
            args.data_path, "experiments", experiment_dir_name, "config"
        )
        os.makedirs(config_path, exist_ok=True)
        grid_df.to_csv(os.path.join(config_path, "config.csv"), index_label="id")

        for i, config_dict in enumerate(the_grid):
            yaml_write(config_dict, os.path.join(config_path, "{}.yaml".format(i)))
