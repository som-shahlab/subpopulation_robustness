import pandas as pd
import numpy as np
import os

# sys.path.append("../")
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from group_robustness_fairness.mimic_eicu.common import (
    save,
    split,
    Discretizer,
    bin_age,
)
from group_robustness_fairness.mimic_eicu.data_extraction_mortality import (
    data_extraction_mortality,
)

benchmark_dir = Path("/scratch/hdd001/projects/ml4h/projects/eICU_Benchmark")
eicu_dir = Path("/scratch/hdd001/projects/ml4h/projects/eicu-crd")
output_dir = Path("/scratch/hdd001/home/haoran/stanford_robustness/data/")
output_dir.mkdir(parents=True, exist_ok=True)

patients = pd.read_csv((eicu_dir / "patient.csv"))[
    ["patientunitstayid", "gender", "age", "ethnicity", "hospitalid"]
]
hospitals = pd.read_csv((eicu_dir / "hospital.csv"))
hospitals["region"] = hospitals["region"].fillna("Missing")

normal_values = {
    "Eyes": 4,
    "GCS Total": 15,
    "Heart Rate": 86,
    "Motor": 6,
    "Invasive BP Diastolic": 56,
    "Invasive BP Systolic": 118,
    "O2 Saturation": 98,
    "Respiratory Rate": 19,
    "Verbal": 5,
    "glucose": 128,
    "admissionweight": 81,
    "Temperature (C)": 36,
    "admissionheight": 170,
    "MAP (mmHg)": 77,
    "pH": 7.4,
    "FiO2": 0.21,
}
cat_variables = ["GCS Total", "Eyes", "Motor", "Verbal"]

# tasks = ['eicu_los', 'eicu_inhospital_mortality' ]
tasks = ["eicu_inhospital_mortality"]

for task in tasks:
    if task == "eicu_inhospital_mortality":
        ts_df = data_extraction_mortality(str(benchmark_dir)).sort_values(
            by=["patientunitstayid", "itemoffset"]
        )
        targets = (
            ts_df.groupby("patientunitstayid")
            .agg({"hospitaldischargestatus": "first"})
            .reset_index()
        )
        pat_df = pd.merge(patients, hospitals, on="hospitalid", how="left")
        pat_df = pd.merge(pat_df, targets, on="patientunitstayid", how="inner").rename(
            columns={"hospitaldischargestatus": "target"}
        )

    pat_df["age"] = pat_df["age"].astype(float)
    pat_df["ethnicity"] = pat_df["ethnicity"].fillna("Other")
    pat_df.loc[
        ~pat_df.ethnicity.isin(["Caucasian", "African American"]), "ethnicity"
    ] = "Other"

    ts_df = ts_df[["patientunitstayid", "itemoffset"] + list(normal_values.keys())]

    pat_df = pat_df[pat_df.patientunitstayid.isin(ts_df.patientunitstayid)]
    assert pat_df.gender.isnull().sum() == 0
    assert pat_df.age.isnull().sum() == 0
    pat_df["age_group"] = pat_df.age.apply(bin_age)

    ts_df = ts_df.merge(
        pat_df[["patientunitstayid", "gender", "age"]],
        on="patientunitstayid",
        how="left",
    ).set_index("patientunitstayid")
    ts_df.gender = ts_df.gender.map({"Male": 1, "Female": 0})

    ts_df["itemoffset"] -= 1
    ts_df = ts_df.rename(columns={"itemoffset": "Hours"})

    possible_values = {}
    for i in cat_variables:
        ts_df[i] = ts_df[i].astype(str)
        possible_values[i] = list(ts_df[i].dropna().unique())

    dis = Discretizer(
        timestep=1.0,
        store_masks=True,
        impute_strategy="previous",
        config_path=os.path.join(os.path.abspath(""), "discretizer_config.json"),
        possible_values=possible_values,
    )

    X_data = []
    for pat in tqdm(pat_df["patientunitstayid"]):
        ex = ts_df.loc[pat]
        if isinstance(ex, pd.Series):
            ex = pd.DataFrame(ex).T
        if ex["Hours"].iloc[0] != 0:
            temp = ex.iloc[0].copy()
            temp["Hours"] = 0
            ex = pd.concat((temp.to_frame().T, ex), ignore_index=True)
        X, col_names = dis.transform(ex.values, header=ts_df.columns, end=47)
        X_data.append(X)

    features = np.stack(X_data)
    pat_df = pat_df.reset_index().rename(columns={"index": "array_index"})
    vocab = col_names.split(",")

    train_val_inds, test_inds = train_test_split(
        pat_df.index, test_size=0.25, random_state=42
    )
    pat_df["fold_id"] = None
    pat_df.loc[train_val_inds, "fold_id"] = "train"
    pat_df.loc[test_inds, "fold_id"] = "test"

    meta = {
        "targets": ["target"],
        "groups": ["gender", "ethnicity", "age_group"],
        "vocab": vocab,
    }

    pat_df = split(pat_df, target_name="target")

    (output_dir / task).mkdir(exist_ok=True)
    save(pat_df, features, vocab, meta, output_dir / task)
