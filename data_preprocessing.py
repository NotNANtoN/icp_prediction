import copy
import os
import sys
from os.path import join

import numpy as np
import pandas as pd

sys.path.insert(0, '')
from src.preprocess_utils.create_corpus_utils import store_df, create_devtest_splits_and_save, \
    create_trainval_splits_and_save
from src.preprocess_utils.preprocessing_utils import apply_yeojohnson, remove_univariate_outliers, \
    remove_multivariate_outliers, normalize
from src.preprocess_utils.missing_utils import fill_missings
from src.preprocess_utils.argparser import create_argparser, check_exp_id
from src.preprocess_utils.preprocess_covid_raw_data import get_and_store_all_data
from src.utils.args import read_args

"""
This script performs all pre-processing steps and allows for selection of applied procedures.
"""


def out_dir_name(yeo, norm_method, fill_method, remove_outliers, remove_multi_outliers):
    """Determine name of output directory with respect to the performed procedures."""
    name = "yeo_" + ("Y" if yeo else "N")
    name = join(name, norm_method)
    name = join(name, fill_method)
    name = join(name, "uni_clip_" + (str(remove_outliers) if remove_outliers else "N"))
    name = join(name, "multi_clip_" + ("Y" if remove_multi_outliers else "N"))
    return name


def _create_dest_dir(outpath, args):
    path = join(outpath, out_dir_name(args.yeo, args.norm_method, args.fill_method, args.remove_outliers,
                                      args.remove_multi_outliers))
    os.makedirs(path, exist_ok=True)
    return path


def _remove_outliers(df, dev_idcs, test_idcs, path, args):
    new_dev_idcs = None

    # Remove outliers if flagged
    if args.remove_outliers:
        df = remove_univariate_outliers(df, args.remove_outliers)

    # Remove multivariate outliers if flagged
    if args.remove_multi_outliers:
        dev_df, test_df = df.iloc[dev_idcs], df.iloc[test_idcs]
        # Only apply to dev set
        dev_df, outlier_idcs = remove_multivariate_outliers(dev_df)

        # Store outlier idcs wrt. original df for replicability
        if args.save:
            np.save(os.path.join(path, "outlier_idcs"), outlier_idcs)
        # Reset index and get dev_idcs wrt. new df
        dev_df = dev_df.reset_index(drop=True)
        print(dev_df)
        new_dev_idcs = dev_df.index
        # Merge back the reduced dev set and the original test set
        df = pd.concat([dev_df, test_df], axis=0, ignore_index=True, sort=False)

    return df, new_dev_idcs


def _prep_data_get_devset(df, outpath):
    # Drop cases where the targets are missing
    print("Dev and test splits:")
    dev_idcs, test_idcs = create_devtest_splits_and_save(df, outpath)
    return df, dev_idcs, test_idcs


def preprocess(df, outpath, dev_idcs, test_idcs, args):
    """Do all preprocessing steps.

    Args:
    -------
        df (pandas.DataFrame) : raw data
        outpath (path-like): destination of the preprocessed data
        args (Namespace): commandline input
    """

    print(f"\n\n New pre-processing...\n\n\t\t Args: {args} \n\n")
    # Create destination directory
    path = _create_dest_dir(outpath, args)
    os.makedirs(path, exist_ok=True)

    no_var_cols = df.loc[:, [col for col in df.columns if df[col].std() == 0]]
    assert len(no_var_cols.columns) == 0, no_var_cols.columns

    # Apply Yeo-Johnson transformation if flagged
    if args.yeo:
        df = apply_yeojohnson(df)

    no_var_cols = df.loc[:, [col for col in df.columns if df[col].std() == 0]]
    assert len(no_var_cols.columns) == 0, no_var_cols.columns

    # Normalization:
    df = normalize(df, method=args.norm_method)

    # Fill NaNs:
    df = fill_missings(df, args.fill_method, args.estimator)

    # Outlier detection & handling per dataframe
    df, new_dev_idcs = _remove_outliers(df, dev_idcs, test_idcs, path, args)

    # Store full dataset w/o split/k-fold (multi-out removed here if applicable)
    if args.save:
        store_df(df, path)

    # Continue only with dev split
    if new_dev_idcs is not None:
        dev_idcs = new_dev_idcs
    dev_df = df.iloc[dev_idcs]
    # Store respective dev_idcs in the subfolder (if no cases removed, should be equal to the idcs in the head folder)
    if args.save:
        np.save(os.path.join(path, "dev_idcs"), dev_idcs)

    # Split dev set into different train and val sets and store them
    print("Train and val splits:")
    if args.save:
        create_trainval_splits_and_save(dev_df, path)
        
    print("Total size: ", df.shape)
    print("Dev size: ", dev_df.shape)
    print("\n\n ... finished pre-processing")


def setup_and_start_preprocessing(passed_args=None):
    parser = create_argparser()
    args = read_args(parser, passed_args)
    args = check_exp_id(args)

    # Setting in and output paths
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #proj_root = os.path.abspath(join(dir_path, os.pardir))
    data_path = "data"

    # Get dataset
    df_raw = get_and_store_all_data(data_path, join("src", "preprocess_utils"))
    df, dev_idcs, test_idcs = _prep_data_get_devset(df_raw, data_path)

    # Start pre-processing
    if args.all:
        print("Starting pre-processing of all combinations of settings.")
        for yeo in [0, 1]:
            for norm in ["z", "minmax"]:
                for fill in ["median", "minus", "iterative"]:
                    for uni in [0.95, 0.999, 0.9999]:
                        for multi in [0, 1]:
                            args.yeo = yeo
                            args.norm_method = norm
                            args.fill_method = fill
                            args.remove_outliers = uni
                            args.remove_multi_outliers = multi
                            preprocess(copy.deepcopy(df), data_path, dev_idcs, test_idcs, args)
    else:
        print("Starting pre-processing with the following settings: \n\t{}{}{}{}{}\n"
              .format(("- Yeo Johnson\n\t" if args.yeo else ""),
                      "- Normalization/Standardization with " + args.norm_method + "\n\t",
                      "- Fill missings through {}{}\n\t".format(args.fill_method,
                                                                (f" with {args.estimator} estimator"
                                                                 if args.fill_method == 'iterative' else "")),
                      (f"- Remove outliers with {args.remove_outliers} quantile\n\t" if args.remove_outliers else ""),
                      ("- Apply IsolationForest for multivariate outlier removal\n" if args.remove_multi_outliers
                       else "")))
        preprocess(df, data_path, dev_idcs, test_idcs, args)
    return True


if __name__ == "__main__":
    setup_and_start_preprocessing()
