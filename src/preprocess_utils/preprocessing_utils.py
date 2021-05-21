import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import yeojohnson
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

import src.constants as constants

"""Utilities for pre-processing."""


def apply_yeojohnson(df):
    # Remove col names that should not be yeo-johnsoned:
    col_names = list(df.columns)
    # - remove categorical features from list
    no_yeo_categorical = _get_categorical_cols(df)
    # - remove features that were before shown to map to one single value after yeo
    ignore_list_path = os.path.join('src', 'preprocess_utils', 'feature_lists', 'ignorelist_yeo.npy')
    if os.path.isfile(ignore_list_path):
        no_yeo_bad_transform = np.load(ignore_list_path)
        apply_yeo = list(set(col_names) - set(no_yeo_categorical) - set(no_yeo_bad_transform))
    else:
        apply_yeo = list(set(col_names) - set(no_yeo_categorical))

    count = 0
    for col_name in apply_yeo:
        if df[col_name].dtype in ['float64', 'float32']:
            col = df[col_name].dropna()
            # col_min = col.min()
            # if col_min <= 0:
            #    col += abs(col.min()) + 0.001
            transformed_array, _ = yeojohnson(col)
            transformed_series = df[col_name].copy()
            transformed_series[~transformed_series.isna()] = transformed_array
            # Overwrite col in df only if yeo outcome is okay
            if _yeo_integrity_okay(transformed_series, col_name):
                df[col_name] = transformed_series
                count += 1
        else:
            print(f"Warning - {col_name} is no float")
    print(f"Applied yeo to {count} out of {len(col_names)} features\n")
    return df


def _yeo_integrity_okay(transformed_series, col_name):
    # Check if yeo transform creates infinite values
    if transformed_series.isin([np.inf, -np.inf]).sum() > 0:
        print(f"Warning - Yeo transformation created >=1 infinite value in column {col_name} - reverting transform")
        return False
    # Check if feature gets mapped to one single value
    if len(transformed_series.unique()) == 1:
        print(f"Warning - Yeo transformation mapped all values of {col_name} to one value - reverting transform")
        return False
    # Check if now all data is NaN
    if transformed_series.isna().sum().sum() == len(transformed_series):
        print(f"Warning - Column {col_name} is all NaN after Yeo transformation - reverting transform")
        return False
    if transformed_series.std() == 0:
        print(f"Warning - Column {col_name} has an std of 0!")
        return False
    return True


def _get_categorical_cols(df):
    cat_cols = df.select_dtypes(include=['category']).columns
    return cat_cols


def _clip(series, count_dict, quantile):
    higher_bound = series.quantile(quantile)
    lower_bound = series.quantile(1 - quantile)
    count_dict["above_bound"] += len(series[series > higher_bound])
    count_dict["below_bound"] += len(series[series < lower_bound])
    series[series > higher_bound] = higher_bound
    series[series < lower_bound] = lower_bound
    return series


def remove_univariate_outliers(df, quantile):
    count_dict = {
        "above_bound": 0,
        "below_bound": 0
    }

    if isinstance(constants.gender, list):
        # make sure that the first 2 values are m, f
        df = df.transform(lambda col: col.groupby(df[constants.gender[0]]).transform(lambda x:
                                                                                   _clip(x, count_dict, quantile)))
    else:
        df = df.transform(lambda col: col.groupby(df[constants.gender]).transform(lambda x: _clip(x,
                                                                                      count_dict, quantile)))

    print("Univariate outlier clipping:")
    print("Clipped", count_dict["above_bound"], " above the bounds and ", count_dict["below_bound"], "below the bounds")
    return df


def remove_multivariate_outliers(df: pd.DataFrame):
    label_col = constants.label_col
    cols_for_outlier_removal = constants.cols_for_outlier_removal

    clf = IsolationForest(random_state=0, contamination=0.02, n_estimators=100)
    out_inliers = clf.fit_predict(df.loc[:, cols_for_outlier_removal])
    outlier_mask = (out_inliers == -1)
    # Only non-target cases should be treated as outliers:
    outlier_mask *= ~(df[label_col].astype(np.bool))
    # Get idcs:
    inlier_idcs = np.where(~outlier_mask)
    outlier_idcs = np.where(outlier_mask)

    # outlier_idcs = df[out_inliers == -1].index  # -1 are outliers and 1 are inliers
    print(f"\nIsolationForest detected {np.sum(out_inliers == -1)} multivariate outliers "
          f"(indices: {list(outlier_idcs)}). Now removing.")
    old_len = len(df)
    # df = df.drop(outlier_idcs)
    outliers = df.iloc[outlier_idcs]
    print(f"Num outliers: {len(df.iloc[outlier_idcs])}"
          f"\nNum {label_col} in outliers: {outliers[label_col].sum()}"
          f"\nFraction {label_col} cases in outliers: {outliers[label_col].mean()}")
    print(f"Fraction {label_col} cases in inliers: {df.iloc[inlier_idcs][label_col].mean()}")
    df = df.iloc[inlier_idcs]
    assert abs(len(df) - old_len) <= 50, f"Removed too many multivariate outliers: {abs(len(df) - old_len)}"

    return df, outlier_idcs


def normalize(df, method="minmax"):
    """Normalizes the data either with Min/Max or z-standardization.
    Param:
        method (str) : Either "minmax" or "z" for the respective normalization method.
    """

    if method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(df)
        df_normalized = scaler.transform(df)

    elif method == "z":
        df_means = df.mean(axis=0)
        df_stds = df.std(axis=0)
        # Set mean of binaries to 0 and std to 1:
        binary_idcs = [i for i, c in enumerate(df.columns) if len(set(df[c].dropna())) == 2]
        df_means[binary_idcs] = 0
        df_stds[binary_idcs] = 1

        std_zero_mask = df_stds == 0
        print("STD of zero: ", std_zero_mask.sum())
        df_stds[std_zero_mask] = 1
        # Apply normalization:
        df_normalized = (df - df_means) / df_stds

    else:
        sys.exit("Abort - Issue with '--norm_method' flag: Invalid normalization method. "
                 "Make sure to provide either 'minmax' or 'z' as an input.")

    # Apply normalization:
    df = pd.DataFrame(df_normalized, columns=df.columns)
    return df


def add_prefix(df, old_names, prefix):
    old_names = [n for n in old_names if n != 'subject']
    new_names = [prefix + n for n in old_names]
    map_names = dict(zip(old_names, new_names))
    df = df.rename(columns=map_names)
    return df


def set_nans(df):
    df = df.replace("#NULL!", math.nan)
    df = df.replace(' ', math.nan)
    df = df.replace('NaN', math.nan)
    df = df.replace(' ', math.nan)
    return df
