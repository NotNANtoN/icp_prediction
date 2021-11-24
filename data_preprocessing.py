import copy
import os
import sys
from os.path import join

import numpy as np
import pandas as pd

sys.path.insert(0, '')
from src.preprocess_utils.create_corpus_utils import create_balanced_split
from src.preprocess_utils.preprocessing_utils import apply_yeojohnson, remove_univariate_outliers, \
    remove_multivariate_outliers, normalize, create_missing_feat_list
from src.preprocess_utils.missing_utils import fill_missings
from src.preprocess_utils.argparser import create_argparser, check_exp_id
from src.preprocess_utils.preprocess_icp import get_and_store_all_data
from src.utils.args import read_args
import src.constants as constants


"""
This script performs all pre-processing steps and allows for selection of applied procedures.
"""


def out_dir_name(minutes, yeo, norm_method, fill_method, remove_outliers, remove_multi_outliers):
    """Determine name of output directory with respect to the performed procedures."""
    name = str(minutes) + "_"
    name = join(name, "yeo_" + ("Y" if yeo else "N"))
    name = join(name, "normalization_" + (norm_method if norm_method else "None"))
    name = join(name, fill_method)
    name = join(name, "uni_clip_" + (str(remove_outliers) if remove_outliers else "N"))
    name = join(name, "multi_clip_" + ("Y" if remove_multi_outliers else "N"))
    return name


def _create_dest_dir(outpath, args):
    path = join(outpath, out_dir_name(args.minutes, args.yeo, args.norm_method, args.fill_method, args.remove_outliers,
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


def preprocess(inputs, targets, outpath, dev_idcs, test_idcs, args):
    """Do all preprocessing steps.

    Args:
    -------
        outpath (path-like): destination of the preprocessed data
        args (Namespace): commandline input
    """

    print(f"\n\n New pre-processing...\n\n\t\t Args: {args} \n\n")
    # Create destination directory
    path = _create_dest_dir(outpath, args)
    os.makedirs(path, exist_ok=True)

    # Apply Yeo-Johnson transformation if flagged
    if args.yeo:
        df = apply_yeojohnson(df)

    # Normalization:
    if args.norm_method is not None: 
        inputs = normalize(inputs, method=args.norm_method)

    fill_nans = False
    
    import src.constants as constants
    label_col = constants.label_col
    
    if fill_nans:
        # Fill NaNs:

        # TODO: Fix merging and separating of targets and inputs! Atm the df is just doubled in length insted of adding the proper columns

        targets = targets[label_col]
        inputs[label_col] = targets.to_list()
        #df = pd.concat([inputs, targets])
        df = fill_missings(inputs, args.fill_method, args.estimator)

        # Outlier detection & handling per dataframe
        inputs, targets = inputs.drop(columns=[label_col]), inputs[label_col]
        inputs, new_dev_idcs = _remove_outliers(inputs, dev_idcs, test_idcs, path, args)

        #df = pd.concat([inputs, targets])
        inputs[label_col] = targets.to_list()
        print(df)
    else:
        print(inputs.shape)
        print(targets.shape)
        df = pd.concat([inputs, targets], axis=1).drop_duplicates()
        print(df.shape)
    
    create_balanced_split(df, path, [3, 5])
    
    return
    
    
    # Store full dataset w/o split/k-fold (multi-out removed here if applicable)
    if args.save:
        store_df(df, path)
        # store all nan-features (includes "don't know")
        df, missingness_features = create_missing_feat_list(df)
        # save names of input features & missingness features
        input_features = [c for c in df.columns if
                          c != constants.label_col and c not in constants.compound_label_cols_incl_diagnosed]
        for name, l in zip(["input_features", "missing_feat_names"],
                           [input_features, missingness_features]):
            np.save(os.path.join(path, name), l)

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
    parser.add_argument("--minutes", default=60)
    args = read_args(parser, passed_args)
    args = check_exp_id(args)

    # Setting in and output paths
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #proj_root = os.path.abspath(join(dir_path, os.pardir))
    data_path = "data"

    minutes = args.minutes
    norm = "yeojo" # yeojo, z or minmax possible
    
    # stays constant (only varied for phase)
    hour = 0
    
    # Get dataset
    inputs, targets = get_and_store_all_data(data_path, minutes, norm, hour)
    #dev_idcs, test_idcs = create_devtest_splits_and_save(inputs, data_path)

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
                            preprocess(copy.deepcopy(inputs), copy.deepcopy(targets), data_path, dev_idcs, test_idcs, args)
    else:
        print("Starting pre-processing with the following settings: \n\t{}{}{}{}{}\n"
              .format(("- Yeo Johnson\n\t" if args.yeo else ""),
                      (f"- Normalization/Standardization with {args.norm_method}\n\t" if
                      args.norm_method else ""),
                      ("- Fill missings through {}{}\n\t".format(args.fill_method,
                                                                (f" with {args.estimator} estimator"
                                                                 if args.fill_method ==
                                                                    'iterative' else ""))),
                      (f"- Remove outliers with {args.remove_outliers} quantile\n\t" if args.remove_outliers else ""),
                      ("- Apply IsolationForest for multivariate outlier removal\n" if args.remove_multi_outliers
                       else "")))
        preprocess(copy.deepcopy(inputs), copy.deepcopy(targets), data_path, None, None, args)
    return True


if __name__ == "__main__":
    setup_and_start_preprocessing()
