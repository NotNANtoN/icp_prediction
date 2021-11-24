import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.constants as constants


def minmax(a):
    a_min = a.min()
    return (a - a_min) / (a.max() - a_min)


def _one_hot(df, columns):
    for c in columns:
        if c not in df.columns:
            # print(f"{c} not in columns")
            break
        dummies = pd.get_dummies(df[c])
        # last dummy category is "don't know" -> mark with "_nan"
        dummies.columns = [
            f"{c}_{int(val)}_nan" if i == len(dummies.columns) - 1 else f"{c}_{int(val)}" for i, val
            in enumerate(sorted(dummies.columns))]
        # also if nan, set last dummy to 1 (= "weiß nicht/ kA")
        if df[c].isna().sum() > 0:
            dummies.loc[df[c].isna(), dummies.columns[-1]] = 1
        df = df.drop(c, axis=1)
        df = pd.concat([df, dummies], axis=1)
    return df


def _single_val_to_bin(df):
    # Some questions only had 1 as response, else nan
    for c in df.columns:
        if len(set(df[c].dropna())) == 1:
            df[c] = df[c].fillna(0)
    return df


def _load_data(minutes):    
    path = f"data/df_final_{minutes}.csv"
    print("Reading Werte df: ", path)
    df = pd.read_csv(path)
    print("Data loaded!")
    
    
    print("Start ICU cleaning")
    
    
    target_name = "ICP_Vital"
   
    # Check input sizes and target sizes
    #print("Name Min Max Mean PercMissing")
    #for name in df.columns:
    #    col = df[name]
    #    print(name, round(col.min(), 3), round(col.max(), 3), round(col.mean(), 3), round(col.isna().mean(), 3))
        
    # Separate in inputs and targets
    inputs = df.drop(columns=[target_name])
    targets = df[target_name]
    
    
    pat_data_list = [df[df["Pat_ID"] == pat] for pat in df["Pat_ID"].unique()]
    icps = [pat[target_name][~pat[target_name].isna()] for pat in pat_data_list]
    for icp in icps:
        if math.isnan(icp.mean()):
            print(icp)
    icps = [i.mean() for i in icps]
    #print("mean icps: ", icps)
    assert not math.isnan(np.mean(icps))
    
    return inputs, targets


def clip_quantile(arr, quant_val):
    return np.clip(arr, np.quantile(arr, 1 - quant_val), np.quantile(arr, quant_val))


def _handle_low_std_variables(df, delete, threshold=0.01):
    print("Num cols before std filter:", len(df.columns))
    low_var_found = [c for c in df.columns if minmax(clip_quantile(df[c], 0.99)).std() < threshold]
    if len(low_var_found) == 0:
        return df
    print("Std threshold: ", threshold)
    #print(f"The following variables have a std below {threshold}:\n{low_var_found}")
    for c in low_var_found:
        if delete and c != constants.label_col:
            print(f"Removing {c} with std {minmax(df[c]).std()} and value counts: \n{df[c].value_counts()}")
            df = df.drop(c, axis=1)
    print("Num cols after std filter:", len(df.columns))
    return df


def remove_correlations(dataset, targets, threshold):
    # remove target from this correlation table as we want to keep features that are highly correlated with the target
    print("Cols before corr:", len(dataset.columns))
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    print("Deleting column with name, ", colname, " as it is too highly correlated (", corr_matrix.iloc[i, j], ") with ", corr_matrix.columns[j])
                    del dataset[colname] # deleting the column from the dataset                
    print("Cols after corr:", len(dataset.columns))
    # check correlations with Vital_ICP value
    dataset["ICP"] = list(targets)
    corr_matrix = dataset.corr()
    icp_cor = corr_matrix["ICP"].to_numpy()
    thresh = 0.0001
    too_small_corr_mask = np.abs(icp_cor) < thresh
    drop_cols = ["ICP"] + list(np.array(corr_matrix.columns)[too_small_corr_mask])
    drop_cols = [c for c in drop_cols if "DB_" not in c]
    print("Dropping because of too small correlation with target: ", drop_cols)
    dataset = dataset.drop(columns=drop_cols)        
    
    return dataset


def get_and_store_all_data(data_dir, minutes, norm, hour):
    # define path
    root_folder = 'data'
    minutes = str(minutes)
    #dataset_name = "Datenbank_Werte.csv"  #f'Datenbank_{norm}_{minutes}min_{hour}h.csv'  
    #load_path = os.path.join(root_folder, dataset_name)
    # load df
    inputs, targets = _load_data(minutes)
    
    
    # Identify Nans:
    mask = (inputs == np.inf) + (inputs == -np.inf)
    print("Total infs: ", mask.to_numpy().sum())
    inputs = inputs.replace([np.inf, -np.inf], np.nan)
    
    print("Number of cols at start: ", len(inputs.columns))
    
    # some values are 1 or missing -> set missing to 0
    inputs = _single_val_to_bin(inputs)

    # one-hot encode selected variables
    for l in []:
        inputs = _one_hot(inputs, l)

    # remove variables with especially low standard deviation
    #inputs = _handle_low_std_variables(inputs, delete=True, threshold=0.025)
    
    # check missingness - not necessary anymore, is done in notebook
    #for col in inputs.columns:
    #    missingness_frac = inputs[col].isna().mean()
    #    if missingness_frac > 0.99:
    #        print(col, " has ", missingness_frac, " NaNs and is therefore removed")
    #        inputs = inputs.drop(columns=[col])
             
    # check correlations
    inputs = remove_correlations(inputs, targets, 0.99)
    
    print("Inputs shape: ", inputs.shape)
    print("Inputs columns: ", inputs.columns)
    print("Targets: ", targets)
    
    return inputs, targets



