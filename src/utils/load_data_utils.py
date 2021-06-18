import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, train_test_split

import src.constants as constants


def read_covid_data(name, v=1, miss_feats=0, features=None):
    wd = os.getcwd()
    if "pi_corona" in wd.split("/")[-1]:
        base_path = "data/"
    elif "src" in wd.split("/")[-1]:
        base_path = "../data/"
    else:
        base_path = "../../data/"

    path = base_path + name + "/"
    # Get data
    # Load from disk:
    try:
        df = pd.read_pickle(os.path.join(path, "df.pkl"))
    except ValueError:
        df = pd.read_csv(os.path.join(path, "df.csv"))
    # with open(os.path.join(path, "df.pkl")) as f:
    #    df = pickle.load(f)

    missing_feat_names = np.load(path + "missing_feat_names.npy")
    input_features = np.load(path + "input_features.npy")

    if features is not None:
        # If using preselected set of features (e.g. results from RFE), select only those:
        selected_features = joblib.load(features)
        assert isinstance(selected_features, list), "Selected features must provided as list of strings ..."
        df = df[selected_features + [constants.label_col]]
    else:
        df = df[list(input_features) + [constants.label_col]]
        if not miss_feats:
            drop_cols = [col for col in missing_feat_names if col in df.columns]
            df = df.drop(columns=drop_cols)
    assert len(df) > 0, "No data left..."
    
    # delete rows where all currently used datatypes were missing
    #df = _drop_fully_missing_cases(df, blood, clinical, imaging, imaging_pca, sparse_img, path,
    # v=v)

    if v:
        print("Feature names: ", list(df.columns))
    return df, path


def _drop_fully_missing_cases(df, blood, clinical, imaging, imaging_pca, sparse_img, path, v=0):
    if not(blood or clinical or imaging or imaging_pca or sparse_img):
        return df
    masks = []
    datatype_names = ["blood", "clinical", "imaging", "imaging", "imaging"]
    for i, datatype in enumerate([blood, clinical, imaging, imaging_pca, sparse_img]):
        if datatype:
            name = datatype_names[i]
            mask = np.load(os.path.join(path, f"{name}_names_empty_cases_mask.npy"))
            masks += [mask]

    combined_mask = np.logical_and(*masks) if len(masks) > 1 else masks[0]
    assert len(combined_mask) == len(df), f'{len(combined_mask)}  {len(df)}'
    non_empty_indcs = np.where(~combined_mask)
    df = df.iloc[non_empty_indcs]
    if v:
        print(f"\n{sum(combined_mask)} rows were all nan and deleted. {len(df)} cases remain.")
    return df


def read_mock_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df = (df - df.mean()) / df.std()
    df["Target"] = data.target
    return df


def get_mock_idcs(df, nf, split):
    if split == 'no-split':
        return None, None, None, None
    # Dev/Test split
    dev_data, test_data, dev_idcs, test_idcs = train_test_split(df, np.arange(len(df)), test_size=0.25,
                                                                stratify=df["Target"], random_state=42)
    # Train/Val splits
    if nf > 1:
        skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
        folds = skf.split(dev_data, dev_data["Target"])
        splits = [fold for fold in folds]
        train_idcs = [split[0] for split in splits]
        val_idcs = [split[1] for split in splits]
    elif split == 'train/val' or 'train/test':
        # give np.arange(len(dev_data)) to get the associated idcs of the same split
        train_data, val_data, train_idcs, val_idcs = train_test_split(dev_data, np.arange(len(dev_data)),
                                                                      test_size=0.25,
                                                                      stratify=df.iloc[dev_idcs]["Target"],
                                                                      random_state=42)
    else:
        raise ValueError("Unknown split option: " + str(split))

    return dev_idcs, test_idcs, train_idcs, val_idcs, dev_data


def remove_max_idcs_to_fit_df(df, idcs_list):
    """Given a list of splitting idcs and a dataframe, remove the maximal idcs of the splitting idx lists as long as
    the maximum index of all lists is too large to index the df"""
    while max(*[max(idx_list) for idx_list in idcs_list]) + 1 - len(df):
        max_idcs = [np.argmax(idx_list) for idx_list in idcs_list]
        max_vals = [idcs_list[idx][max_idx] for idx, max_idx in enumerate(max_idcs)]
        idx_of_max_val = np.argmax(max_vals)
        del idcs_list[idx_of_max_val][max_idcs[idx_of_max_val]]


def get_target_idcs_from_df(df, path, nf):
    dev_idcs = list(np.load(path + "dev_idcs.npy"))
    test_idcs = [i for i in range(len(df)) if i not in set(dev_idcs)]
    remove_max_idcs_to_fit_df(df, [dev_idcs, test_idcs])

    dev_df = df.iloc[dev_idcs]
    train_idcs = list(np.load(path + "train_idcs.npy"))
    val_idcs = [i for i in range(len(dev_df)) if i not in set(train_idcs)]
    remove_max_idcs_to_fit_df(dev_df, [train_idcs, val_idcs])

    if nf:
        train_idcs = []
        val_idcs = []
        for i in range(nf):
            split_path = os.path.join(path, f'{str(nf)}_folds', f'{i}.npy')
            split_val_idcs = np.load(split_path)
            split_train_idcs = [i for i in range(len(dev_idcs)) if i not in set(split_val_idcs)]
            train_idcs.append(split_train_idcs)
            val_idcs.append(split_val_idcs)
        remove_max_idcs_to_fit_df(dev_df, [*train_idcs, *val_idcs])

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
    df, path = read_covid_data(df_path, v=v, **read_data_args)
    dev_idcs, test_idcs, train_idcs, val_idcs, dev_df = get_target_idcs_from_df(df, path, nf)
    return dev_idcs, dev_df[constants.label_col]


def load_data(df_name, split, nf, v=1, dev_idcs=None, test_idcs=None, train_idcs=None,
              val_idcs=None,
              **read_data_args):
    """Input is the name of the specific folder within .data/.
    """
    # Read data from file and get splitting indices:
    if df_name == "mock":
        df = read_mock_data()
        dev_idcs_loaded, test_idcs_loaded, train_idcs_loaded, val_idcs_loaded, dev_df = get_mock_idcs(df, nf, split)
    else:
        df, path = read_covid_data(df_name, v=v, **read_data_args)
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


def _check_array(array):
    if isinstance(array, list):
        for subarray in array:
            _check_array(subarray)
    else:
        array = np.array(array).astype(float)
        assert np.sum(np.isinf(array)) + np.sum(np.isnan(array)) == 0


def get_data(*read_data_args, **read_data_kwargs):
    data = load_data(*read_data_args, **read_data_kwargs)
    x_train, y_train, x_eval, y_eval, n_features, feature_names, class_weights = data
    _check_array(x_train)
    _check_array(x_eval)
    _check_array(y_train)
    _check_array(y_eval)
    return data
