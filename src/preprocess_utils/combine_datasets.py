import os

import numpy as np
import pandas as pd

from src.preprocess_utils.blood_data_preprocessing import load_blood_data
from src.preprocess_utils.clinical_data_preprocessing import load_clinical_data
from src.preprocess_utils.imaging_data_preprocessing import load_imaging_data, reduce_imaging, load_sparse_imaging_data
from src.preprocess_utils.missing_utils import drop_empty_target


def _load_all_data(data_path, lists_path, remove_precipitals=True):
    imaging_df_sparse = load_sparse_imaging_data(data_path, lists_path)

    # Get pre-cleaned DataFrames
    blood_df = load_blood_data(data_path, lists_path)
    imaging_df = load_imaging_data(data_path, lists_path)
    clinical_df = load_clinical_data(data_path, lists_path, remove_precipitals=remove_precipitals)
    # Get dim-reduced dfs
    imaging_df_reduced_0_8 = reduce_imaging(imaging_df, 0.8)
    imaging_df_reduced_0_99 = reduce_imaging(imaging_df, 0.99)
    # New imaging df:

    return [blood_df, clinical_df, imaging_df, imaging_df_reduced_0_8, imaging_df_reduced_0_99, imaging_df_sparse]


def _generate_missing_row_masks(df, zipped_lists, lists_path):
    name_list_dir = os.path.join(lists_path, "feature_lists", "name_lists")
    os.makedirs(name_list_dir, exist_ok=True)
        
    lists, names = zip(*zipped_lists)
    for name in ["blood_names", "imaging_names", "clinical_names"]:
        columns = lists[names.index(name)]
        # take only _nan cols
        columns = [col for col in columns if col.endswith('_nan')]
        # check if all nan columns are coded 1
        bool_list = df[columns].apply(lambda x: all(x), axis=1)
        print(f"For dataype {name}, {sum(bool_list)} rows are fully missing.")
        #df[f"{name}_all_na"] = bool_list
        assert len(bool_list) == len(df)
        np.save(os.path.join(name_list_dir, f"{name}_empty_cases_mask.npy"), np.array(bool_list))
        
    return df


def _generate_name_lists(df, lists_path):
    # Extract outcomes:
    df_no_outcomes = df.drop(columns=["POD", "POCD"])
    # Extract inputs separately:
    missing_feat_names = []
    blood_names = []
    imaging_names = []
    clinical_names = []
    imagingpca_names_0_8 = []
    imagingpca_names_0_99 = []
    static_names = []
    sparse_img_names = []

    for col in df_no_outcomes:
        print(col)
        # Nan column?
        if "_nan" in col:
            missing_feat_names.append(col)
        # Which datatype?
        if "blood_" in col:
            blood_names.append(col)
        elif "sparse_img" in col:
            sparse_img_names.append(col)
        elif "imaging_" in col:
            imaging_names.append(col)
        elif "imagingpca_0.8_" in col:
            imagingpca_names_0_8.append(col)
        elif "imagingpca_0.99_" in col:
            imagingpca_names_0_99.append(col)
        elif "clinical_" in col:
            clinical_names.append(col)
        else:
            static_names.append(col)
    assert len(static_names) == 4
    # Save names:
    lists = [missing_feat_names, blood_names, imaging_names, clinical_names, static_names, imagingpca_names_0_8,
             imagingpca_names_0_99, sparse_img_names]
    names = ["missing_feat_names", "blood_names", "imaging_names", "clinical_names", "static_names",
             "imagingpca_names_0_8", 'imagingpca_names_0_99', 'sparse_img_names']

    # list: mask for missing cases
    name_list_dir = os.path.join(lists_path, "feature_lists", "name_lists")
    os.makedirs(name_list_dir, exist_ok=True)
    for list_, name_ in zip(lists, names):
        np.save(os.path.join(name_list_dir, name_), list_)

    return zip(lists, names)


def get_and_store_all_data(data_path, lists_path, remove_precipitals=True):
    """Loads the separate data types, triggers their individual pre-processing and returns a merged DataFrame.
    
    Args: 
        data_path : Path to the file within which all core_data_sets (.csv) reside
        
    Returns:
        merged_df : DataFrame that combines all cleaned data sets (a raw version is also stored to data_path).
    """
    df_list = _load_all_data(data_path, lists_path, remove_precipitals=remove_precipitals)

    # Merge along subject ID
    merged_df = pd.concat(df_list, axis=1, join='outer', ignore_index=False)
    merged_df.to_csv(os.path.join(data_path, 'all_data_raw.csv'))
    #merged_df = merged_df.drop('subject', axis=1)
    merged_df = drop_empty_target(merged_df)

    # Store names
    zipped_lists_names = _generate_name_lists(merged_df, lists_path)
    merged_df = _generate_missing_row_masks(merged_df, zipped_lists_names, lists_path)

    print(f"\nMerged all data - num of cases: {len(merged_df)}, num of features: {len(merged_df.columns)}")
    return merged_df
