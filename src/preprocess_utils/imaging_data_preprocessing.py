import os
import json

import numpy as np
import pandas as pd
import sklearn

from src.preprocess_utils.preprocessing_utils import add_prefix, set_nans

paper_rename_img = {
    'BrainVol_cm3_pre': 'Brain volume (cm3)',
    # 'BFCS_Vol_cm3_pre': 'Basal forebrain cholinergic system volume (mm3)',
    'NBM_Vol_mm3_pre': 'Nucleus basalis Meynert volume (mm3)',
    'Hippocampus': 'Hippocampus volume (mm3)'
}
json.dump(paper_rename_img, open("data/renaming_for_paper_imaging.json", 'w+'))


def _clean_imaging_columns(lists_path, df):
    df = set_nans(df)
    # Neuroimaging data features with some unnecessary features already removed
    allowlist = np.load(os.path.join(lists_path, "feature_lists", "allowlist_imaging_data.npy"))
    df = df[allowlist]
    # Typecast all feature columns to numeric
    df.loc[:, df.columns != 'subject'] = df.loc[:, df.columns != 'subject'].apply(pd.to_numeric, errors='coerce')
    return df


def _add_miss_features_for_imaging_data(lists_path, df):
    # these have systematically different amounts of missings such that we handle those features separately
    cubic_vol_features = np.load(os.path.join(lists_path, "feature_lists", "cubic_features_imaging_data.npy"))
    region_features = list(set(df.columns) - set(cubic_vol_features))

    # Create new columns for the two cases, set it to 1 if all values in that row are NaN
    df['imaging_regions_nan'] = np.where(df[region_features].sum(axis=1) == 0, 1, 0)
    df['imaging_cubic_vol_nan'] = np.where(df[cubic_vol_features].sum(axis=1) == 0, 1, 0)
    return df


def load_imaging_data(path, lists_path):
    imaging_df = pd.read_csv(os.path.join(path, "core_data_set_20200211_adalab_imaging.csv"))

    # Clean columns for imaging dataframe
    imaging_df = _clean_imaging_columns(lists_path, imaging_df)

    # Creates one missingness feature for all brain region features 
    imaging_df = _add_miss_features_for_imaging_data(lists_path, imaging_df)

    # Add prefix in order to identify the brain imaging data later
    imaging_df = add_prefix(imaging_df, imaging_df.columns, 'imaging_')

    return imaging_df


def load_sparse_imaging_data(path, lists_path):
    img_df = pd.read_excel(os.path.join(path, "Core_Data_Set_12-06-2020.xlsx"))#, encoding="ISO-8859-1")
    # Load hippocampal volume from extra file
    hippocampal_df = pd.read_excel(os.path.join(path, 'core_data_set_20200211_adalab_imaging_hippocampal_vol.xls'))
    hippocampal_df = hippocampal_df.rename(columns={"Patient_ID": "vs0040_v1"})
    img_df = img_df.merge(hippocampal_df, how='left')
    # Drop non-imaging cols
    img_cols = ['BrainVol_cm3_pre', 'BFCS_Vol_mm3_pre', 'NBM_Vol_mm3_pre', 'Hippocampus']
    non_img_cols = [col for col in img_df if col not in img_cols]
    img_df = img_df.drop(columns=non_img_cols)

    # Remove unneeded region
    img_df = img_df.drop(columns='BFCS_Vol_mm3_pre')
    # Rename:
    # rename cols
    paper_rename_img = json.load(open("data/renaming_for_paper_imaging_lammers.json"))
    img_df = img_df.rename(columns=paper_rename_img)


    print(img_df.columns)
    img_df.columns = ['sparse_img_' + col for col in img_df.columns]
    return img_df


def reduce_imaging(df, variance_threshold):
    # Apply dimension reduction
    # create reducer
    reducer = sklearn.decomposition.PCA(n_components=variance_threshold)
    # separate df into two parts
    data = df.to_numpy()
    # remove missingness features
    data = data[:, :-2]

    # Indices that are only existant in one group of patients or only in small groups: 18, 62, 661
    # Hence we seem to only have a large group of patients that have all measurements - that's all
    filter_idcs = [18, 62, 661]
    keep_idcs = [i for i in range(data.shape[1]) if i not in filter_idcs]
    data = data[:, keep_idcs]

    nan = np.isnan(data)
    groups = {}
    for idx, pat in enumerate(nan):
        if tuple(pat) in groups:
            groups[tuple(pat)].append(idx)
        else:
            groups[tuple(pat)] = [idx]

    assert len(groups) == 2, "There should only be one group with only missing features and one without missingness."
    full_group = None
    # Get "full_group", meaning data points with no missing imaging data
    for key in groups:
        if sum(key) != len(key):
            full_group = groups[key]
    full_data = data[full_group]
    assert np.isnan(full_data).sum() == 0
    # Z-standardize PCA input
    means = full_data.mean(axis=0)
    stds = full_data.std(axis=0)
    full_data = (full_data - means) / stds

    # Reduce data for cases that have imaging data
    reduced_full_data = reducer.fit_transform(full_data)
    dims = reduced_full_data.shape[1]

    # Create new data array filled with NaNs for all patients
    new_data = np.zeros((len(df), dims)) * np.nan
    # Fill reduced data in correct order
    new_data[full_group] = reduced_full_data
    # Create new missingness features
    missingness_col = np.ones((len(df), 1))
    missingness_col[full_group] = 0
    # Add missingness features
    new_data = np.concatenate([new_data, missingness_col], axis=1)
    new_columns = [f'imagingpca_{variance_threshold}_{i}' for i in range(dims)] + \
                  [f'imagingpca_{variance_threshold}_nan']
    df = pd.DataFrame(data=new_data, columns=new_columns)

    return df
