from __future__ import absolute_import
from __future__ import print_function


import os
import group_robustness_fairness.mimic_eicu.utils as utils

import pandas as pd


def data_extraction_mortality(root_dir, time_window=48):
    # all_df = utils.embedding(root_dir)
    all_df = pd.read_csv(os.path.join(root_dir, "all_data.csv"))
    all_mort = utils.filter_mortality_data(all_df)
    all_mort = all_mort[all_mort["itemoffset"] <= time_window]
    return all_mort
