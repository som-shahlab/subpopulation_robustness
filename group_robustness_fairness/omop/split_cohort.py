import pandas as pd
import argparse
from group_robustness_fairness.prediction_utils.omop.util import patient_split_cv

parser = argparse.ArgumentParser()

parser.add_argument(
    "--source_cohort_path", type=str, help="The source file", required=True,
)

parser.add_argument(
    "--target_cohort_path", type=str, help="The destination file", required=True,
)

parser.add_argument(
    "--nfold", type=int, default=5,
)

parser.add_argument(
    "--test_frac", type=float, default=0.25,
)

parser.add_argument(
    "--seed", type=int, default=9526,
)

if __name__ == "__main__":
    args = parser.parse_args()

    cohort = pd.read_parquet(args.source_cohort_path)
    if "fold_id" in cohort.columns:
        cohort = cohort.drop(columns="fold_id")
    split_cohort = patient_split_cv(
        cohort,
        test_frac=args.test_frac,
        eval_frac=float((1 - (args.test_frac)) / (args.nfold + 1)),
        nfold=args.nfold,
        seed=args.seed,
    )
    split_cohort.to_parquet(args.target_cohort_path, index=False)
