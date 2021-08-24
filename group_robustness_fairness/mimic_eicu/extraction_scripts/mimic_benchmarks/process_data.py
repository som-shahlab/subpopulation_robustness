import sys

sys.path.append("../")
from group_robustness_fairness.mimic_eicu.common import (
    save,
    split,
    Discretizer,
    bin_age,
)
import pandas as pd
import numpy as np
from group_robustness_fairness.mimic_eicu.readers import (
    InHospitalMortalityReader,
    DecompensationReader,
    LengthOfStayReader,
)
from pathlib import Path
import os
from tqdm.auto import tqdm
import pickle
from sklearn.model_selection import StratifiedKFold
import re


def clean_eth(string):
    string = string.lower()
    if bool(re.search("^white", string)):
        return "white"
    elif bool(re.search("^black", string)):
        return "black"
    else:
        return "other"


mimic_benchmark_dir = Path("/scratch/hdd001/projects/ml4h/projects/MIMIC_benchmarks/")
output_dir = Path("/scratch/hdd001/home/haoran/stanford_robustness/data/")
output_dir.mkdir(parents=True, exist_ok=True)


readers = {
    "mimic_inhospital_mortality": [
        InHospitalMortalityReader(
            dataset_dir=mimic_benchmark_dir / "in-hospital-mortality" / "train"
        ),
        InHospitalMortalityReader(
            dataset_dir=mimic_benchmark_dir / "in-hospital-mortality" / "test"
        ),
    ],
}


def process_targets(task, y, series):
    if task in ["mimic_inhospital_mortality"]:
        series["target"] = y
    else:
        raise NotImplementedError


all_stays = pd.read_csv(
    os.path.join(mimic_benchmark_dir, "root/", "all_stays.csv"), parse_dates=["INTIME"]
).set_index("ICUSTAY_ID")

for task in readers:
    train_reader = readers[task][0]
    test_reader = readers[task][1]
    df_data = []
    X_data = []
    dis = Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        config_path=os.path.join(os.path.dirname(__file__), "discretizer_config.json"),
    )

    for fold in ["train", "test"]:
        reader = train_reader if fold == "train" else test_reader
        for i in tqdm(range(reader.get_number_of_examples())):
            ex = reader.read_example(i)
            subj_id = int(ex["name"].split("_")[0])
            stay = pd.read_csv(
                os.path.join(
                    mimic_benchmark_dir,
                    "root",
                    fold,
                    str(subj_id),
                    ex["name"].split("_")[1] + ".csv",
                )
            ).iloc[0]
            icustay_id = int(stay["Icustay"])
            series = {}
            series["ID"] = ex["name"][:-4]
            series["Gender"] = all_stays.loc[icustay_id, "GENDER"]
            series["Ethnicity"] = all_stays.loc[icustay_id, "ETHNICITY"]
            series["Age"] = float(stay["Age"])
            series["fold_id"] = fold
            process_targets(task, ex["y"], series)

            ex["header"] += ["Gender", "Age"]
            gender_cat = 1.0 if series["Gender"] == "M" else 0.0
            n_t = ex["X"].shape[0]

            # add two additional static features to the time series
            first_row = (
                ["0.0"]
                + [""] * (ex["X"].shape[1] - 1)
                + [str(gender_cat), str(series["Age"])]
            )
            ex["X"] = np.concatenate(
                (ex["X"], np.empty([ex["X"].shape[0], 2], dtype="<U18")), axis=-1
            )
            ex["X"] = np.concatenate(
                (np.expand_dims(np.array(first_row), 0), ex["X"]), axis=0
            )
            X, col_names = dis.transform(ex["X"], header=ex["header"], end=48)
            X_data.append(X)
            df_data.append(series)

    features = np.stack(X_data)
    df = (
        pd.DataFrame.from_dict(df_data)
        .reset_index()
        .rename(
            columns={
                "index": "array_index",
                "Ethnicity": "ethnicity",
                "Gender": "gender",
            }
        )
    )
    vocab = col_names.split(",")

    meta = {
        "targets": ["target"],
        "groups": ["gender", "ethnicity", "age_group"],
        "vocab": vocab,
    }
    df = split(df)
    df["ethnicity"] = df["ethnicity"].map(clean_eth)
    df["age_group"] = df["Age"].map(bin_age)

    (output_dir / task).mkdir(exist_ok=True)
    save(df, features, vocab, meta, output_dir / task)
