import numpy as np
import pickle
from pathlib import Path
import copy
import pandas as pd
from itertools import chain
import yaml


def combinations(grid):
    keys = list(grid.keys())
    limits = [len(grid[i]) for i in keys]
    all_args = []

    index = [0] * len(keys)

    while True:
        args = {}
        for c, i in enumerate(index):
            key = keys[c]
            args[key] = grid[key][i]
        all_args.append(args)

        # increment index by 1
        carry = False
        index[-1] += 1
        ind = len(index) - 1
        while ind >= 0:
            if carry:
                index[ind] += 1

            if index[ind] == limits[ind]:
                index[ind] = 0
                carry = True
            else:
                carry = False
            ind -= 1

        if carry:
            break

    return all_args


def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].hparams()


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname


def flatten(lst):
    return list(chain(*lst))


common_grid = {
    "lr": [1e-4, 1e-5],
    "num_hidden": [1, 3],
    "hidden_dim": [128, 256],
    "fold_id": list(map(str, range(5))),
    "early_stopping": [True],
    "early_stopping_patience": [25],
    "model_type": ["MLP", "GRU"],
    "num_epoch": [150],
    "label_col": [],  # populated
    "eval_attributes": [],  # populated
    "run_evaluation": [True],
    "save_outputs": [True],
    "run_evaluation_group": [True],
    "run_evaluation_group_standard": [True],
    "normalize_data": [True],
    "drop_prob": [0.25, 0.75],
    "batch_size": [512],
}

erm_group_aware = {
    "group_objective_type": ["standard"],
    "sensitive_attribute": [],  # populated
    "balance_groups": [True, False],
    "selection_metric": ["loss", "auc_min", "loss_bce_max"],
    "run_evaluation_group_fair_ova": [True],
}

erm_subset = {  # populate
    "group_objective_type": ["standard"],
    "subset_attribute": [],
}

dro_tuning = {
    "lr_lambda": [1.0, 1e-1, 1e-2],
    "group_objective_type": ["dro"],
    "sensitive_attribute": [],  # populated
    "group_objective_metric": ["loss", "baselined_loss", "auc_proxy"],
    "balance_groups": [True, False],
    "selection_metric": ["loss", "auc_min", "loss_bce_max"],
    "run_evaluation_group_fair_ova": [True],
}

dro_tuning_adjusted = {
    "lr_lambda": [1.0, 1e-1, 1e-2],
    "group_objective_type": ["dro"],
    "sensitive_attribute": [],  # populated
    "group_objective_metric": ["size_adjusted_loss", "size_adjusted_loss_reciprocal"],
    "adjustment_scale": [0.01, 0.1, 1],
    "balance_groups": [True, False],
    "selection_metric": ["loss", "auc_min", "loss_bce_max"],
    "run_evaluation_group_fair_ova": [True],
}

bootstrap = {
    "data_path": [],  # populated
    "eval_attributes": [],  # populated,
    "task": [],  # populated
    "result_path": ["/scratch/hdd001/home/haoran/stanford_robustness/results"],
}


def read_meta(data_path):
    return pickle.load((Path(data_path) / "meta.pkl").open("rb"))


class BaseExpERM:
    fname = "train_model.py"

    @staticmethod
    def populate_grid(cls, data_path):
        common_grid["data_path"] = [data_path]
        meta = read_meta(data_path)
        common_grid["label_col"] = meta["targets"]
        common_grid["eval_attributes"] = [meta["groups"]]
        erm_group_aware["sensitive_attribute"] = meta["groups"]
        # dro_tuning['sensitive_attribute'] = meta['groups']
        # dro_tuning_adjusted['sensitive_attribute'] = meta['groups']

        if cls.flat:
            common_grid["model_type"] = ["MLP"]  # no GRU since not time series
            common_grid["normalize_data"] = [False]

    @staticmethod
    def get_erm_subset(data_path):  # returns list of dictionaries
        def populate(subset_dict, grp, uniques):
            subset_dict = copy.deepcopy(subset_dict)
            subset_dict["subset_attribute"] = [grp]
            subset_dict["subset_group"] = uniques
            return subset_dict

        meta = read_meta(data_path)
        cohort = pd.read_pickle(Path(data_path) / "cohort.pkl")
        return [
            populate(erm_subset, i, list(cohort[i].unique())) for i in meta["groups"]
        ]

    @classmethod
    def hparams(cls):
        BaseExpERM.populate_grid(
            cls, cls.data_path
        )  # populates common, ERM group aware
        return combinations({**common_grid, **erm_group_aware}) + flatten(
            list(
                map(
                    lambda x: combinations({**common_grid, **x}),
                    BaseExpERM.get_erm_subset(cls.data_path),
                )
            )
        )


class BaseExpDRO:
    fname = "train_model.py"

    @staticmethod
    def populate_grid(cls, data_path, config_path):
        common_grid["data_path"] = [data_path]
        meta = read_meta(data_path)
        config = yaml.safe_load(open(config_path, "r"))

        common_grid["label_col"] = [config["task"][config["task"].find("_") + 1 :]]

        common_grid["eval_attributes"] = [meta["groups"]]
        dro_tuning["sensitive_attribute"] = meta["groups"]
        dro_tuning_adjusted["sensitive_attribute"] = meta["groups"]

        if cls.flat:
            common_grid["model_type"] = ["MLP"]  # no GRU since not time series
            common_grid["normalize_data"] = [False]

        for hp in config:
            if hp != "task":
                common_grid[hp] = [config[hp]]

    @classmethod
    def hparams(cls):
        BaseExpDRO.populate_grid(
            cls, cls.data_path, cls.config_path
        )  # populates common and DRO
        return combinations({**common_grid, **dro_tuning}) + combinations(
            {**common_grid, **dro_tuning_adjusted}
        )


data_paths = {
    "MIMICMortality": "/scratch/hdd001/home/haoran/stanford_robustness/data/mimic_inhospital_mortality",
    "eICUMortality": "/scratch/hdd001/home/haoran/stanford_robustness/data/eicu_inhospital_mortality",
}

#### write experiments here
# ERM Experiments
class MIMICMortalityERM(BaseExpERM):
    data_path = data_paths["MIMICMortality"]
    flat = False


class eICUMortalityERM(BaseExpERM):
    data_path = data_paths["eICUMortality"]
    flat = False


# DRO Experiments
class MIMICMortalityDRO(BaseExpDRO):
    data_path = data_paths["MIMICMortality"]
    config_path = "/scratch/hdd001/home/haoran/stanford_robustness/results/MIMICMortality_target_erm_config.yaml"
    flat = False


class eICUMortalityDRO(BaseExpDRO):
    data_path = data_paths["eICUMortality"]
    config_path = "/scratch/hdd001/home/haoran/stanford_robustness/results/eICUMortality_target_erm_config.yaml"
    flat = False


class Bootstrap:
    fname = "bootstrapping_replicates.py"

    @classmethod
    def hparams(cls):
        commands = []
        tasks = ["MIMICMortality_target", "eICUMortality_target"]
        config_file = pd.read_csv(
            Path(bootstrap["result_path"][0]) / "selected_configs.csv"
        )
        for task in tasks:
            bootstrap_i = copy.deepcopy(bootstrap)
            data_path = data_paths[task[: task.find("_")]]
            bootstrap_i["data_path"] = [data_path]
            meta = read_meta(data_path)
            bootstrap_i["eval_attributes"] = [meta["groups"]]
            bootstrap_i["task"] = [task]
            commands.extend(combinations(bootstrap_i))
        assert len(commands) == len(tasks)
        return commands
