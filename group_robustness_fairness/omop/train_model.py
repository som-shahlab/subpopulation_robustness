import numpy as np
import pandas as pd
import os
import joblib
import configargparse as argparse
import copy
import torch
import logging
import sys


from group_robustness_fairness.prediction_utils.pytorch_utils.models import (
    FixedWidthModel,
)
from group_robustness_fairness.prediction_utils.pytorch_utils.datasets import (
    ArrayLoaderGenerator,
)

from group_robustness_fairness.prediction_utils.util import yaml_write
from group_robustness_fairness.prediction_utils.pytorch_utils.robustness import (
    group_robust_model,
)

from group_robustness_fairness.prediction_utils.pytorch_utils.metrics import (
    StandardEvaluator,
)

parser = argparse.ArgumentParser(config_file_parser_class=argparse.YAMLConfigFileParser)

parser.add_argument("--config_path", required=False, is_config_file=True)

# Path configuration
parser.add_argument(
    "--data_path", type=str, default="", help="The root path where data is stored",
)

parser.add_argument(
    "--features_path", type=str, default="", help="The root path where data is stored",
)

parser.add_argument(
    "--cohort_path",
    type=str,
    default="",
    help="File name for the file containing metadata",
)

parser.add_argument(
    "--vocab_path",
    type=str,
    default="",
    help="File name for the file containing feature information",
)

parser.add_argument(
    "--features_row_id_map_path",
    type=str,
    default="",
    help="Maps identifiers in the cohort dataframe to rows of the feature matrix",
)

parser.add_argument(
    "--logging_path", type=str, default=None, help="A path to store logs",
)

parser.add_argument(
    "--result_path", type=str, required=True, help="A path where results will be stored"
)

parser.add_argument(
    "--weight_var_name",
    type=str,
    default=None,
    help="The name of a column with weights",
)

# Hyperparameters - training dynamics
parser.add_argument(
    "--num_epochs", type=int, default=10, help="The number of epochs of training"
)

parser.add_argument(
    "--iters_per_epoch",
    type=int,
    default=100,
    help="The number of batches to run per epoch",
)

parser.add_argument("--batch_size", type=int, default=256, help="The batch size")

parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")

parser.add_argument(
    "--gamma", type=float, default=0.95, help="Learning rate decay (exponential)"
)

parser.add_argument(
    "--early_stopping",
    dest="early_stopping",
    action="store_true",
    help="Whether to use early stopping",
)

parser.add_argument("--early_stopping_patience", type=int, default=5)

parser.add_argument(
    "--selection_metric",
    type=str,
    default="loss",
    help="The metric to use for model selection",
)

# Hyperparameters - model architecture
parser.add_argument(
    "--num_hidden", type=int, default=1, help="The number of hidden layers"
)

parser.add_argument(
    "--hidden_dim", type=int, default=128, help="The dimension of the hidden layers"
)

parser.add_argument(
    "--normalize", dest="normalize", action="store_true", help="Use layer normalization"
)

parser.add_argument(
    "--drop_prob", type=float, default=0.25, help="The dropout probability"
)

parser.add_argument("--weight_decay", type=float, default=0)

# Experiment configuration
parser.add_argument(
    "--fold_id", type=str, default="1", help="The fold id to use for early stopping"
)

parser.add_argument(
    "--label_col",
    type=str,
    default="LOS_7",
    help="The name of a column in cohort to use as the label",
)

parser.add_argument(
    "--data_mode", type=str, default="array", help="Which mode of source data to use",
)

parser.add_argument("--sparse_mode", type=str, default="list", help="The sparse mode")

parser.add_argument(
    "--num_workers",
    type=int,
    default=5,
    help="The number of workers to use for data loading during training",
)

parser.add_argument(
    "--deterministic",
    dest="deterministic",
    action="store_true",
    help="Whether to use deterministic training",
)

parser.add_argument(
    "--seed", type=int, default=2020, help="The seed",
)

parser.add_argument(
    "--cuda_device", type=int, default=0, help="The cuda device id",
)

parser.add_argument(
    "--logging_metrics",
    type=str,
    nargs="*",
    required=False,
    default=["auc", "loss_bce"],
    help="metrics to use for logging during training",
)

parser.add_argument(
    "--logging_threshold_metrics",
    type=str,
    nargs="*",
    required=False,
    default=None,
    help="threshold metrics to use for logging during training",
)

parser.add_argument(
    "--logging_thresholds",
    type=str,
    nargs="*",
    required=False,
    default=None,
    help="thresholds to use for threshold-based logging metrics",
)

parser.add_argument(
    "--eval_attributes",
    type=str,
    nargs="+",
    required=False,
    default=None,
    help="The attributes to use to perform a stratified evaluation. Refers to columns in the cohort dataframe",
)

parser.add_argument(
    "--eval_thresholds",
    type=float,
    nargs="+",
    required=False,
    default=None,
    help="The thresholds to apply for threshold-based evaluation metrics",
)

parser.add_argument(
    "--sample_keys", type=str, nargs="*", required=False, default=None, help=""
)

parser.add_argument(
    "--replicate_id", type=str, default="", help="Optional replicate id"
)


## Arguments related to group fairness and robustness
parser.add_argument(
    "--sensitive_attribute",
    type=str,
    default=None,
    help="The attribute to be fair with respect to",
)

parser.add_argument(
    "--balance_groups",
    dest="balance_groups",
    action="store_true",
    help="Whether to rebalance the data so that data from each group is sampled with equal probability",
)

parser.add_argument(
    "--group_objective_type",
    type=str,
    default="standard",
    help="""
    Options:
        standard: train with a standard ERM objective over the entire dataset
        dro: train with a practical group distributionally robust strategy
    """,
)

parser.add_argument(
    "--group_objective_metric",
    type=str,
    default="loss",
    help="""The metric used to construct the group objective. 
    Refer to the implementation of group_robust_model""",
)

## Arguments specific to DRO
parser.add_argument(
    "--lr_lambda", type=float, default=1e-2, help="The learning rate for DRO"
)

parser.add_argument(
    "--update_lambda_on_val",
    dest="update_lambda_on_val",
    action="store_true",
    help="Whether to update the lambdas on the validation set",
)

parser.add_argument("--adjustment_scale", type=float, default=None)

# Args related to subsetting data
parser.add_argument("--subset_attribute", type=str, default=None)
parser.add_argument("--subset_group", type=str, default=None)

# Boolean arguments
parser.add_argument(
    "--save_outputs",
    dest="save_outputs",
    action="store_true",
    help="Whether to save the outputs of evaluation",
)

parser.add_argument(
    "--logging_evaluate_by_group",
    dest="logging_evaluate_by_group",
    action="store_true",
    help="Whether to evaluate the model for each group during training",
)

parser.add_argument(
    "--run_evaluation",
    dest="run_evaluation",
    action="store_true",
    help="Whether to evaluate the trained model",
)

parser.add_argument(
    "--run_evaluation_group",
    dest="run_evaluation_group",
    action="store_true",
    help="Whether to evaluate the trained model for each group",
)

parser.add_argument(
    "--run_evaluation_group_standard",
    dest="run_evaluation_group_standard",
    action="store_true",
    help="Whether to evaluate the model with the standard evaluator",
)

parser.add_argument(
    "--print_debug",
    dest="print_debug",
    action="store_true",
    help="Whether to print debugging information",
)

parser.add_argument(
    "--disable_metric_logging",
    dest="disable_metric_logging",
    action="store_true",
    help="Whether to disable metric logging during training",
)

parser.add_argument(
    "--overwrite",
    dest="overwrite",
    action="store_true",
    help="If false, will skip training if result files exists",
)

parser.set_defaults(
    normalize=False,
    early_stopping=False,
    save_outputs=False,
    balance_groups=False,
    logging_evaluate_by_group=False,
    run_evaluation=True,
    run_evaluation_group=False,
    run_evaluation_group_standard=False,
    run_evaluation_group_fair_ova=False,
    spectral_norm=False,
    deterministic=True,
    update_lambda_on_val=False,
    print_debug=False,
    disable_metric_logging=False,
    overwrite=False,
)


def filter_cohort(cohort, subset_attribute=None, subset_group=None):
    # Global filter
    if "gender_concept_name" in cohort.columns:
        cohort = cohort.query('gender_concept_name != "No matching concept"')

    # Custom filter
    if (subset_attribute is not None) and (subset_group is not None):
        if not (subset_attribute in cohort.columns):
            raise ValueError("subset_attribute not in cohort columns")
        cohort = cohort.query(
            "{subset_attribute} == '{subset_group}'".format(
                subset_attribute=subset_attribute, subset_group=subset_group
            )
        )
    return cohort


def read_file(filename, columns=None, **kwargs):
    load_extension = os.path.splitext(filename)[-1]
    if load_extension == ".parquet":
        return pd.read_parquet(filename, columns=columns, **kwargs)
    elif load_extension == ".csv":
        return pd.read_csv(filename, usecols=columns, **kwargs)


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)
    if (not args.overwrite) and os.path.exists(
        os.path.join(args.result_path, "result_df_group_standard_eval.parquet")
    ):
        print(
            "Result exists at {}".format(
                os.path.join(args.result_path, "result_df_group_standard_eval.parquet")
            )
        )
        sys.exit(0)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.deterministic:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config_dict = copy.deepcopy(args.__dict__)

    if args.weight_var_name is not None:
        config_dict["weighted_loss"] = True
        config_dict["weighted_evaluation"] = True

    logger = logging.getLogger(__name__)
    if config_dict.get("logging_path") is not None:
        logging.basicConfig(
            filename=config_dict.get("logging_path"),
            level="DEBUG" if args.print_debug else "INFO",
            format="%(message)s",
        )
    else:
        logging.basicConfig(
            stream=sys.stdout,
            level="DEBUG" if args.print_debug else "INFO",
            format="%(message)s",
        )

    if args.fold_id == "":
        # To train without a validation set
        train_keys = ["train"]
        eval_keys = ["test"]
    else:
        # To train with a validation set
        train_keys = ["train", "val"]
        eval_keys = ["val", "test"]

    eval_keys = eval_keys + ["eval"]
    config_dict["fold_id_test"] = ["eval", "test"]

    vocab = read_file(args.vocab_path, engine="pyarrow")
    config_dict["input_dim"] = vocab.col_id.max() + 1

    cohort = read_file(args.cohort_path)
    print(args.cohort_path)
    cohort = filter_cohort(
        cohort, subset_attribute=args.subset_attribute, subset_group=args.subset_group
    )

    features = joblib.load(args.features_path)
    if args.features_row_id_map_path != "":
        row_id_map = read_file(args.features_row_id_map_path, engine="pyarrow")
        cohort = cohort.merge(row_id_map)
        config_dict["row_id_col"] = "features_row_id"

    logging.info("Result path: {}".format(args.result_path))
    os.makedirs(args.result_path, exist_ok=True)

    if args.sensitive_attribute is not None:
        loader_generator = ArrayLoaderGenerator(
            features=features,
            cohort=cohort,
            include_group_in_dataset=True,
            group_var_name=args.sensitive_attribute,
            **config_dict
        )
        config_dict["num_groups"] = loader_generator.config_dict["num_groups"]
        config_dict["group_mapper"] = loader_generator.config_dict["group_mapper"]

    else:
        loader_generator = ArrayLoaderGenerator(
            features=features, cohort=cohort, **config_dict
        )

    if args.group_objective_type == "standard":
        model_class = FixedWidthModel
    else:
        assert args.sensitive_attribute is not None
        if args.group_objective_type == "dro":
            model_class = group_robust_model(config_dict["group_objective_metric"])
        else:
            raise ValueError("group_objective_type not defined")

    if (
        args.group_objective_type == "dro"
        and "size_adjusted" in args.group_objective_metric
    ):
        model = model_class(
            cohort_df=cohort.merge(config_dict["group_mapper"]).query(
                "fold_id not in @eval_keys"
            ),
            **config_dict
        )
    else:
        model = model_class(**config_dict)

    logging.info(model.config_dict)

    # Write the resulting config
    yaml_write(config_dict, os.path.join(args.result_path, "config.yaml"))

    loaders = loader_generator.init_loaders(sample_keys=args.sample_keys)

    result_df = model.train(loaders, phases=train_keys)["performance"]
    del loaders

    # Dump training results to disk
    result_df.to_parquet(
        os.path.join(args.result_path, "result_df_training.parquet"),
        index=False,
        engine="pyarrow",
    )

    if args.run_evaluation:
        logging.info("Evaluating model")
        loaders_predict = loader_generator.init_loaders_predict()
        predict_dict = model.predict(loaders_predict, phases=eval_keys)
        del loaders_predict
        output_df_eval, result_df_eval = (
            predict_dict["outputs"],
            predict_dict["performance"],
        )
        logging.info(result_df_eval)

        # Dump evaluation result to disk
        result_df_eval.to_parquet(
            os.path.join(args.result_path, "result_df_training_eval.parquet"),
            index=False,
            engine="pyarrow",
        )
        if args.save_outputs:
            output_df_eval.to_parquet(
                os.path.join(args.result_path, "output_df.parquet"),
                index=False,
                engine="pyarrow",
            )
        if args.run_evaluation_group:
            logging.info("Running evaluation on groups")
            if args.eval_attributes is None:
                raise ValueError(
                    "If using run_evaluation_group, must specify eval_attributes"
                )

            strata_vars = ["phase", "task", "sensitive_attribute", "eval_attribute"]

            output_df_eval = output_df_eval.assign(task=args.label_col)

            output_df_eval = output_df_eval.merge(
                row_id_map, left_on="row_id", right_on="features_row_id"
            ).merge(cohort)

            assert (
                len(set(["eval_attribute", "eval_group"]) & set(output_df_eval.columns))
                == 0
            )
            if args.subset_attribute is not None:
                eval_attributes = args.subset_attribute
            elif args.sensitive_attribute is not None:
                eval_attributes = args.sensitive_attribute
            else:
                eval_attributes = args.eval_attributes

            if isinstance(eval_attributes, str):
                eval_attributes = [eval_attributes]

            output_df_long = output_df_eval.melt(
                id_vars=set(output_df_eval.columns) - set(eval_attributes),
                value_vars=eval_attributes,
                var_name="eval_attribute",
                value_name="eval_group",
            )

            if args.run_evaluation_group_standard:
                # (TODO) add command line arguments to specify evaluation metrics
                evaluator = StandardEvaluator(thresholds=args.eval_thresholds)
                result_df_group_standard_eval = evaluator.evaluate(
                    output_df_long,
                    strata_vars=strata_vars + ["eval_group"],
                    weight_var=config_dict.get("weight_var_name"),
                )
                logging.info(result_df_group_standard_eval)
                result_df_group_standard_eval.to_parquet(
                    os.path.join(
                        args.result_path, "result_df_group_standard_eval.parquet"
                    ),
                    engine="pyarrow",
                    index=False,
                )
