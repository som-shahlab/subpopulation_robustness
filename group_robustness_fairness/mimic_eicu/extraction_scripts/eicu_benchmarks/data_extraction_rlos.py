from __future__ import absolute_import
from __future__ import print_function


import os
import pandas as pd
import group_robustness_fairness.mimic_eicu.utils as utils


def data_extraction_rlos(root_dir):
    # all_df = utils.embedding(root_dir)
    all_df = pd.read_csv(os.path.join(root_dir, "all_data.csv"))
    all_los = utils.filter_rlos_data(all_df)
    return all_los
