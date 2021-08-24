import numpy as np


def normalize(cohort, features, config_dict):
    test_mask = cohort["fold_id"].isin([config_dict["fold_id"], "eval", "test"])
    train_idx = cohort.loc[~test_mask, config_dict["row_id_col"]].values
    test_idx = cohort.loc[test_mask, config_dict["row_id_col"]].values
    reduce_dim = tuple(range(len(features.shape)))[:-1]
    means, stds = (
        np.mean(features[train_idx], axis=reduce_dim),
        np.std(features[train_idx], axis=reduce_dim),
    )

    stds_mod = stds.copy()
    stds_mod[stds_mod == 0] = 1.0
    features = (features - means) / stds_mod
    return features, (means, stds)


def flatten(features, num_incs=4):
    # features: # samples, # channels, # features
    # output: # samples, # features * 4
    x_data = []
    incs = np.linspace(0, features.shape[1], num=num_incs + 1)
    for c, i in enumerate(incs[:-1]):
        x_data.append(features[:, int(i) : int(incs[c + 1]), :].mean(axis=1))
    return np.concatenate(x_data, axis=-1)
