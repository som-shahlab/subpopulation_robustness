from __future__ import absolute_import
from __future__ import print_function


import os
import pandas as pd
import group_robustness_fairness.mimic_eicu.utils as utils


def data_extraction_decompensation(root_dir):
    # all_df = utils.embedding(root_dir)
    all_df = pd.read_csv(os.path.join(root_dir, "all_data.csv"))
    all_dec = utils.filter_decom_data(all_df)
    all_dec = utils.label_decompensation(all_dec)
    return all_dec
