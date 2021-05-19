import os

import numpy as np

from preprocess_utils.create_corpus_utils import create_k_fold, split_df

"""This script creates train and holdout splits as well as optionally k-fold splitting (by default)."""


def create_splits(df, path, kfold, k=None):
    # Create and save train-test indices
    train_idcs, val_idcs = split_df(df, test_size=0.2, val_size=0.2, hard_threshold=0.15, soft_threshold=0.1,
                                    num_allowed=2)
    np.save(os.path.join(path, "train_idcs"), train_idcs)
    np.save(os.path.join(path, "val_idcs"), val_idcs)

    if k is None:
        k = 5
    # Create and save k-fold indices:
    if kfold:
        splits = create_k_fold(df.iloc[train_idcs], k)
        split_path = os.path.join(path, f"{str(k)}_folds")
        os.makedirs(split_path, exist_ok=True)
        for idx, split in enumerate(splits):
            np.save(os.path.join(split_path, str(idx)), split)
