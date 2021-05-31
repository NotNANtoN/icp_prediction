import os
import shutil

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

import src.constants as constants

label_col = constants.label_col
"""
Utilities for the actual corpus creation.
"""


def create_labels(df):
    df_minuses = df.fillna(-1)
    return [df_minuses.iloc[idx][label_col].astype(int).astype(str) for idx in range(len(df))]



def create_balanced_split(df, dev_fraction=0.8, hard_threshold=0.3, soft_threshold=0.2,
                          num_allowed=2, random_state=None):
    eval_fraction = 1 - dev_fraction
    print("Dev/Train size: ", int(dev_fraction * len(df)), "Test/Val size: ", int(eval_fraction * len(df)))
    count = 0
    outliers = num_allowed + 1
    max_diff = hard_threshold + 1
    while outliers > num_allowed or max_diff > hard_threshold:
        # Create split:
        indices = np.arange(len(df))
        labels = create_labels(df)
        dev_data, test_data, dev_idcs, test_idcs = train_test_split(df, indices, test_size=eval_fraction,
                                                                    stratify=labels, random_state=random_state)
        # Test if split is good enough:
        diffs = np.array([0])
        # np.array(np.abs(test_data.mean(axis=0) - dev_data.mean(axis=0))) / np.abs(dev_data.mean(axis=0))
        max_diff = max(diffs)
        # print("test", test_data.mean(axis=0), "dev", dev_data.mean(axis=0))
        # print("first: ", np.abs(test_data.mean(axis=0) - dev_data.mean(axis=0)))
        # print(test_data.mean().mean(), dev_data.mean().mean(), df.mean().mean())
        # print(list(np.round(diffs[diffs > soft_threshold], 2)))
        # print(names[diffs > soft_threshold])
        # print("Mean train data: \n", dev_data.mean(), "Mean test data: \n", test_data.mean())
        # print("Mean deviation: ", np.mean(diffs), "Max deviation:", max_diff)
        outliers = (diffs > soft_threshold).sum()
        count += 1
        if count == 100:
            raise StopIteration("Can't find balanced split")
        print("Num outliers: ", outliers)
        print()
    return dev_idcs, test_idcs


def get_k_fold_indices(train_data, k):
    """Creates a stratified k-fold partition and returns the indices of the evaluation partition"""
    if len(train_data) == k:
        kf = KFold(n_splits=k)
        splits = kf.split(train_data)

    else:
        train_labels = create_labels(train_data)
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        splits = skf.split(train_data, train_labels)
    # Select only validation indices, the others are redundant
    splits = [split[1] for split in splits]
    return splits


def _create_and_store_k_fold(dev_df, k, path):
    if k == "leave_one_out":
        k = len(dev_df)
        folder_name = "leave_one_out/"
    else:
        folder_name = str(k) + "_folds/"
    splits = get_k_fold_indices(dev_df, k)
    # Create k-fold directory
    split_path = os.path.join(path, folder_name)
    os.makedirs(split_path, exist_ok=True)
    # Store eval indices per fold
    for idx, split in enumerate(splits):
        np.save(split_path + str(idx), split)


def create_trainval_splits_and_save(dev_df, path):
    # Apply and store train and validation indices
    train_idcs, val_idcs = create_balanced_split(dev_df)
    np.save(os.path.join(path, "train_idcs"), train_idcs)
    np.save(os.path.join(path, "val_idcs"), val_idcs)

    # Apply and store k-fold
    for k in [2, 3, 4, 5, 10, "leave_one_out"]:
        _create_and_store_k_fold(dev_df, k, path)


def create_devtest_splits_and_save(df, path):
    # Create and store dev & test set idcs
    dev_idcs, test_idcs = create_balanced_split(df)
    np.save(os.path.join(path, "dev_idcs"), dev_idcs)
    np.save(os.path.join(path, "test_idcs"), test_idcs)
    return dev_idcs, test_idcs


def store_df(df, path):
    """Stores a fully processed df (filled NANs etc.)"""
    print(f"Storing data to {path}.\n")

    assert df.isna().to_numpy().sum() == 0

    pd.to_pickle(df, os.path.join(path, "df.pkl"))

    #name_list_dir = os.path.join("src", "preprocess_utils", "feature_lists")
    #for name_list in os.listdir(name_list_dir):
     #   shutil.copy(os.path.join(name_list_dir, name_list), os.path.join(path, name_list))
