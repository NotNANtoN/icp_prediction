import os
import shutil
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import src.constants as constants

label_col_name = constants.label_col
"""
Utilities for the actual corpus creation.
"""


def _create_labels_icp(pat_data_list):
    #mean_short_crit = np.mean([pat["ICP_critical_short"].mean() for pat in pat_data_list])
    #mean_long_crit = np.mean([pat["ICP_critical_long"].mean() for pat in pat_data_list])
    mean_len = np.mean([len(pat) for pat in pat_data_list])
    icps = [pat["ICP_pred"][~pat["ICP_pred"].isna()] for pat in pat_data_list]
    icps = [i.mean() for i in icps]
    mean_icp = np.mean(icps)
    #print("Mean short crit: ", mean_short_crit)
    #print("Mean long crit: ", mean_long_crit)
    print("Mean len: ", mean_len)
    print("Mean icp: ", mean_icp)
    
    labels =  [# (pat_data["Alter"].iloc[0] < mean_age).astype(int).astype(str) +
            # pat_data["Geschlecht"].iloc[0].astype(str) +
            # pat_data["Outcome"].iloc[0].astype(str) +
            (len(pat_data) < mean_len).astype(int).astype(str) +
            ((pat_data["ICP_pred"].mean() < mean_icp).astype(int).astype(str))
            #(pat_data["ICP_critical_short"].mean() < mean_short_crit).astype(int).astype(str) +
            #(pat_data["ICP_critical_long"].mean() < mean_long_crit).astype(int).astype(str)
            for pat_data in pat_data_list]
    return labels


def create_balanced_split(df, path, folds=None):
    """Creates train/val/test splits and stores idcs to path"""
    # store df
    pd.to_pickle(df, os.path.join(path, "df.pkl"))
    df.to_csv(os.path.join(path, "df.csv"))

    
    # Make list out of pat_data_list for splitting:
    if folds is None:
        folds = [5]

    dataset_as_pat_list = [df[df["Pat_ID"] == pat_id] for pat_id in sorted(df["Pat_ID"].unique())]

    # Train-Test split:
    indices = np.arange(len(dataset_as_pat_list))
    labels = _create_labels_icp(dataset_as_pat_list)
    
    import matplotlib.pyplot as plt
    plt.hist(labels)
    plt.savefig("label_hist.pdf")

    dev_data, test_data, dev_idcs, test_idcs = train_test_split(dataset_as_pat_list, indices, test_size=0.2,
                                                                stratify=labels)

    # Train-Val split:
    indices = np.arange(len(dev_data))
    labels = _create_labels_icp(dev_data)
    train_data, val_data, train_idcs, val_idcs = train_test_split(dev_data, indices, test_size=0.2, stratify=labels)

    # K-folds:
    for k in folds:
        create_k_fold(dev_data, labels, k, path)

    # Save train-test indices:
    os.makedirs(path, exist_ok=True)
    np.save(path + "/dev_idcs.pt", dev_idcs)
    np.save(path + "/test_idcs", test_idcs)
    np.save(path + "/val_idcs", val_idcs)


def create_k_fold(dev_data, dev_labels, k, path):
    """Creates a k-fold split and stores the indices in a folder in path"""
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    splits = skf.split(dev_data, dev_labels)
    # splits is list of tuples [(train1, val1), (train2, val2)...]
    splits = [split[1] for split in splits]

    # Save k-fold indices:
    split_path = path + "/" + str(k) + "_folds/"
    os.makedirs(split_path, exist_ok=True)
    for idx, split in enumerate(splits):
        np.save(split_path + str(idx), split)

