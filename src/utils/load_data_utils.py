import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split

import src.constants as constants


def read_icu_data(name, v=1, miss_feats=0):
    # construct base path depedning on current working directoys
    wd = os.getcwd()
    if "icp_prediction" in wd.split("/")[-1]:
        base_path = "data/"
    elif "src" in wd.split("/")[-1]:
        base_path = "../data/"
    else:
        base_path = "../../data/"
    path = base_path + name + "/"
    
    df = pd.read_pickle(os.path.join(path, "df.pkl"))
    
    feat_names = pd.read_pickle(os.path.join(path, "feature_names.pkl"))
    
    dataset_as_pat_list = [df[df["Pat_ID"] == pat_id] for pat_id in sorted(df["Pat_ID"].unique())]

    
    return dataset_as_pat_list, feat_names, path

def remove_max_idcs_to_fit_df(df, idcs_list):
    """Given a list of splitting idcs and a dataframe, remove the maximal idcs of the splitting idx lists as long as
    the maximum index of all lists is too large to index the df"""
    while max(*[max(idx_list) for idx_list in idcs_list]) + 1 - len(df):
        max_idcs = [np.argmax(idx_list) for idx_list in idcs_list]
        max_vals = [idcs_list[idx][max_idx] for idx, max_idx in enumerate(max_idcs)]
        idx_of_max_val = np.argmax(max_vals)
        del idcs_list[idx_of_max_val][max_idcs[idx_of_max_val]]


def get_target_idcs_from_df(df, path, nf):
    test_idcs = list(np.load(path + "test_idcs.npy"))
    dev_idcs = [i for i in range(len(df)) if i not in set(test_idcs)]
    #remove_max_idcs_to_fit_df(df, [dev_idcs, test_idcs])

    dev_df = df[dev_idcs]
    val_idcs = list(np.load(path + "val_idcs.npy"))
    train_idcs = [i for i in range(len(dev_df)) if i not in set(val_idcs)]
    #remove_max_idcs_to_fit_df(dev_df, [train_idcs, val_idcs])

    if nf:
        train_idcs = []
        val_idcs = []
        for i in range(nf):
            split_path = os.path.join(path, f'{str(nf)}_folds', f'{i}.npy')
            split_val_idcs = np.load(split_path)
            split_train_idcs = [i for i in range(len(dev_idcs)) if i not in set(split_val_idcs)]
            train_idcs.append(split_train_idcs)
            val_idcs.append(split_val_idcs)
        #remove_max_idcs_to_fit_df(dev_df, [*train_idcs, *val_idcs])

    return dev_idcs, test_idcs, train_idcs, val_idcs, dev_df


def get_train_eval_data(df, dev_df, test_idcs, train_idcs, val_idcs, nf, split):
    if nf > 1:
        train_data = []
        eval_data = []
        for i in range(nf):
            split_train_idcs = train_idcs[i]
            split_val_idcs = val_idcs[i]
            train_data.append(dev_df.iloc[split_train_idcs])
            eval_data.append(dev_df.iloc[split_val_idcs])
    else:
        if split == "no-split":
            return df, df

        train_str, eval_str = split.split('/')
        # Get data to train on:
        if train_str == "train":
            train_data = dev_df.iloc[train_idcs]
        elif train_str == "dev":
            train_data = dev_df
        else:
            raise ValueError("Unknown split option: " + str(split))
        # Get df to eval on:
        if eval_str == "val":
            eval_data = dev_df.iloc[val_idcs]
        elif eval_str == "test":
            eval_data = df.iloc[test_idcs]
        else:
            raise ValueError("Unknown split option: " + str(split))
    return train_data, eval_data


def input_target_split(df, nf):
    target_names = [constants.label_col]
    drop_names = target_names

    if nf:
        y = [split_df[target_names] for split_df in df]
        x = [split_df.drop(columns=drop_names) for split_df in df]
    else:
        y = [df[target_names]]
        x = [df.drop(columns=drop_names)]
    feature_names = list(x[0].columns)
    # To numpy:
    x = [split.to_numpy().squeeze() for split in x]
    y = [split.to_numpy().squeeze() for split in y]
    return x, y, feature_names


def _calc_class_weights(y):
    mean_label = np.array(y).astype(float).mean()
    class_weights = torch.tensor([mean_label, 1 - mean_label])
    return class_weights


def get_dev_idcs_and_targets(df_path, nf, v=0, **read_data_args):
    list_of_seqs, feat_names, path = read_icu_data(df_path, v=v, **read_data_args)
    dev_idcs, test_idcs, train_idcs, val_idcs, dev_df = get_target_idcs_from_df(list_of_seqs, path, nf)
    return dev_idcs, dev_df[constants.label_col]


def load_data(df_name, split, nf, v=1, dev_idcs=None, test_idcs=None, train_idcs=None,
              val_idcs=None,
              **read_data_args):
    """Input is the name of the specific folder within .data/.
    """
    # Read data from file and get splitting indices:
    df, feat_names, path = read_icu_data(df_name, v=v, **read_data_args)
    dev_idcs_loaded, test_idcs_loaded, train_idcs_loaded, val_idcs_loaded, dev_df = get_target_idcs_from_df(
        df, path, nf)
    # Override train and eval idcs if they are given from the outside:
    if dev_idcs is not None:
        nf = 0
        split = 'dev/test'
        dev_df = df.iloc[dev_idcs]
    else:
        dev_idcs = dev_idcs_loaded
        test_idcs = test_idcs_loaded
    if train_idcs is not None:
        nf = 0
        split = 'train/val'
        dev_df = df
    else:
        train_idcs = train_idcs_loaded
        val_idcs = val_idcs_loaded

    # Get df to train on (apply train/eval splits):
    train_data, eval_data = get_train_eval_data(df, dev_df, test_idcs, train_idcs, val_idcs, nf, split)

    # Extract targets:
    x_train, y_train, feature_names = input_target_split(train_data, nf)
    x_eval, y_eval, feature_names = input_target_split(eval_data, nf)
    n_features = len(feature_names)

    # Get number of features (for neural network creation) and class weights (for weighting the losses):
    class_weights = torch.mean(torch.stack([_calc_class_weights(y_train[i]) for i in range(len(y_train))]), dim=0)

    assert constants.label_col not in feature_names
    if v:
        print('Num input features:\n  ', n_features)

    return x_train, y_train, x_eval, y_eval, n_features, feature_names, class_weights


def _check_array(array, nan_okay=False):
    if isinstance(array, list):
        for subarray in array:
            _check_array(subarray)
    else:
        array = np.array(array).astype(float)
        if not nan_okay:
            assert np.sum(np.isinf(array)) + np.sum(np.isnan(array)) == 0
        else:
            assert np.sum(np.isinf(array)) == 0


def get_data(*read_data_args, **read_data_kwargs):
    data = load_data(*read_data_args, **read_data_kwargs)
    x_train, y_train, x_eval, y_eval, n_features, feature_names = data
    _check_array(x_train)
    _check_array(x_eval)
    _check_array(y_train, nan_okay=True)
    _check_array(y_eval, nan_okay=True)
    return data
