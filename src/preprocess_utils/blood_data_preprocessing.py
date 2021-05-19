import os
import json

import numpy as np
import pandas as pd

from src.preprocess_utils.preprocessing_utils import add_prefix, set_nans
from src.preprocess_utils.plot_utils import make_plots

remove_cols = [#'Troponin', 'NTproBNP', 'MDA', 'Leptin_Lab',
               'Erythroblasten Percent',  # all values are zero
               #'Leptin Lab'
               # 'Zonulin...'

]

paper_rename = {}
# Anstatt CRP HS-CRP (High sensitive) (blutwertnamen)


def _rename(df):
    cols = df.columns
    cols = [col.replace("T1_", "") for col in cols]
    cols = [col[2:] if col.startswith("T1") else col for col in cols]
    cols = [col.replace("T0", "") for col in cols]
    cols = [col.replace("Final", "") for col in cols]
    cols = [col.replace("_", " ") for col in cols]
    cols = [col.replace("mg_dl", "mg/dl") for col in cols]
    cols = [col.replace("pgml", "pg/ml") for col in cols]
    cols = [col.replace("mmolL", "mmol/L") for col in cols]
    cols = [col.replace(" gL", " g/L") for col in cols]
    cols = [col.replace(" mmolmol", " mmol/mol") for col in cols]
    cols = [col.replace("Adj Lab", "") for col in cols]
    cols = [col.replace("Adj Batch", "") for col in cols]
    cols = [col.strip(" ") for col in cols]
    df.columns = cols
    return df


def _clean_blood_columns(df):
    df = _rename(df)

    make_plots(df, prefix='blood')

    print(df.columns)

    # Drop cols
    df = df.drop(columns=remove_cols)
    df = df.drop(columns=['POD'])
    # SORL1 is invalid, remove it
    df = df.drop(columns=["SORL1"])
    # Zonulin was not aim of study, remove it
    df = df.drop(columns=["Zonulin"])

    
    # Reinsert NANs that were incorrectly filled in with zeros
    df = set_nans(df)
    # df.loc[df["IL6"] == 0, "IL6"] = np.nan


    # rename cols
    paper_rename = json.load(open("data/renaming_for_paper_blood_lammers.json"))
    df = df.rename(columns=paper_rename)

    print(df.columns)

    # Typecast all feature columns to float
    for col in df.columns:
        df[col] = df[col].astype(float) if col != 'subject' else df[col]
    return df


def _add_miss_features_for_blood_data(df):
    # Don't encode targets and where there are no missings
    nans = df.isna()#.drop(columns=["subject"])
    for col in nans.columns:
        if nans[col].sum() == 0:
            print("No missings for ", col)
            nans = nans.drop(columns=[col])

    # Missingness features get suffix for identification
    nans = nans.add_suffix("_nan").astype(float)
    df = pd.concat([df, nans], axis=1)
    return df


def load_blood_data(path, miss_feature=True):
    #df = pd.read_excel(os.path.join(path, "Core_Data_Set_12-06-2020.xlsx"))#, encoding="ISO-8859-1")
    df = pd.read_csv(os.path.join(path, "Core_Data_Set_12-06-2020_Added_Labs_LogisticalInfo_BiomarkersAdjForLabs_Corr_NZ.csv"))
    
    # Load T1_Volk_IL8_pgml from extra file
    volk_df = pd.read_excel(os.path.join(path, 'IL8Volk_ID_final070121.xls'))
    volk_df = volk_df.rename(columns={"ID": "vs0040_v1"})
    df = df.merge(volk_df, how='left')

    non_blood_cols = [col for col in df if not col.startswith("T1")]
    #non_blood_cols.remove('Final_T1_TP42_40')
    non_blood_cols.remove('POD')  # is dropped in cleaning
    blood_df = df.drop(columns=non_blood_cols)
    #blood_df = pd.read_csv(os.path.join(path, "core_data_set_20200211_adalab_blood.csv"))


    # Clean columns for blood dataframe
    blood_df = _clean_blood_columns(blood_df)

    # Create missingness features for all columns
    if miss_feature:
        blood_df = _add_miss_features_for_blood_data(blood_df)

    # Add prefix in order to identify the blood data later
    blood_df = add_prefix(blood_df, blood_df.columns, 'blood_')

    return blood_df
