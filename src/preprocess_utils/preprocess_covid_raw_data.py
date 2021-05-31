import os

import numpy as np
import pandas as pd

import src.constants as constants


def _dontknow_to_mean(df, columns):
    # Replace "don't know" with mean
    for c in columns:
        df[c] = df[c].replace({max(df[c]) : round(df[c].mean())})
    return df


def _dontknow_to_lowest(df, columns):
    # Replace "don't know" with lowest value (corresponding e.g. to "no")
    for c in columns:
        df[c] = df[c].replace({max(df[c]) : min(df[c])})
    return df


def _replace_nan_placeholder(df, text_to_val_df, columns, minus_one=False):
    # Replace 99, 999 etc. by np.nan
    for c in columns:
        if not minus_one:
            nan_placeholder = text_to_val_df.loc[text_to_val_df['Name'] == c, 'Value'].to_list()[0]
        else:
            nan_placeholder = -1.0
        if c in df.columns:
            df[c] = df[c].replace(nan_placeholder, np.nan)
    return df


def _check_only_nvars_have_nan(df):
    # Check if remaining nans are only in the interval questions (start with 'n')
    okay = True
    for c in df.columns:
        if df[c].isna().sum() > 0 and not c.startswith('n'):
            okay = False
    return okay


def _create_compound_label(df, columns):
    pos_class_idcs = df.index[df[columns].any(axis=1)].tolist()
    df[constants.label_col] = np.zeros(len(df))
    df.loc[pos_class_idcs, constants.label_col] = 1
    df[constants.label_col] = df[constants.label_col].astype(int)
    print("Target values", set(df[constants.label_col]))
    print(f"Number of cases with positive target {len(pos_class_idcs)}")
    return df


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


def _convert_categorical_to_float(df):
    columns = df.select_dtypes(include=['category']).columns
    for c in columns:
        df[c] = df[c].astype(float)
    return df


def _single_val_to_bin(df):
    # Some questions only had 1 as response, else nan
    for c in df.columns:
        if len(set(df[c].dropna())) == 1:
            df[c] = df[c].fillna(0)
    return df


def _load_variable_values_df(path):
    text_to_val_df = pd.read_excel(path, names=["Name", "Value", "Text"], header=1)
    # Fill all names
    for i in range(len(text_to_val_df)):
        if pd.isna(text_to_val_df.loc[i, "Name"]):
            text_to_val_df.loc[i, "Name"] = text_to_val_df.loc[i - 1, "Name"]
    # make values ints
    for i in range(len(text_to_val_df)):
        val = text_to_val_df.loc[i, "Value"]
        if val == ",00":
            val = 0
        elif val == "1,00":
            val = 1
        else:
            val = str(val).split(",")[0].replace(",", "")
            if val == "":
                print(i)
                val = -1
            else:
                val = int(val)
        text_to_val_df.loc[i, "Value"] = val
    return text_to_val_df


def _load_data(path, text_to_val_df):
    df = pd.read_spss(path)
    # replace strings by values using Variablenwerte.xls
    replace_dict = {name: {row["Text"]: row["Value"] for _, row in
                           text_to_val_df[text_to_val_df["Name"] == name].iterrows()} for name in
                    text_to_val_df["Name"].unique()}
    df = df.replace(replace_dict)
    # replace empty rows by NaN
    df = df.replace({"": np.nan, " ": np.nan})
    # remove "offen" fields
    df = df[[col for col in df if "offen" not in col]]
    df = df.drop(constants.open_ended, axis=1)
    return df


def _handle_low_std_variables(df, delete, threshold=0.01):
    low_var_found = [c for c in df.columns if df[c].std() < 0.01]
    if len(low_var_found) == 0:
        return df
    #print(f"The following variables have a std below {threshold}:\n{low_var_found}")
    for c in low_var_found:
        if delete and c != constants.label_col:
                print(f"Removing {c} with value counts: \n{df[c].value_counts()}")
                df = df.drop(c, axis=1)
    return df


def get_and_store_all_data(data_dir, lists_path):
    variable_names_path = os.path.join(data_dir, "Variablenwerte.xls")
    data_path = os.path.join(data_dir, "f20.0251z_290620.sav")

    text_to_val_df = _load_variable_values_df(variable_names_path)
    df = _load_data(data_path, text_to_val_df)
    # set "don't know" responses to mean where plausible
    df = _dontknow_to_mean(df, constants.ordinal_questions)
    # set "don't know" responses to lowest value where plausible
    df = _dontknow_to_lowest(df, constants.preconditions_when)
    # replace the nan-placeholders by np.nan (can be 99, 999, 9999, 99999, -1 - to my knowledge)
    df = _replace_nan_placeholder(df, text_to_val_df, constants.interval_questions)
    df = _replace_nan_placeholder(df, text_to_val_df, constants.minus_nan, minus_one=True)
    # convert all the categoricals to float
    df = _convert_categorical_to_float(df)
    # some values are 1 or missing -> set missing to 0
    df = _single_val_to_bin(df)

    # one-hot encode selected variables
    for l in [constants.to_one_hot, constants.expect_change, constants.reduced_income,
              constants.age_kids, constants.not_always_applicable]:
        df = _one_hot(df, l)

    # drop some variables or ordinal categories (like some response options of the target questions)
    df = df.drop(constants.drop_variables_list, axis=1)

    # only the numeric/interval variables should have nans now bc. for every other question
    # they're encoded
    assert _check_only_nvars_have_nan(df)

    # remove variables with especially low standard deviation
    df = _handle_low_std_variables(df, delete=False, threshold=0.02)
    # set non_categorical variables to float
    df[constants.non_categorical] = df[constants.non_categorical].apply(lambda c: c.astype(float))
    # create label based on aggregate of constants.compound_label_cols_only_tested
    df = _create_compound_label(df, constants.compound_label_cols_only_tested)
    return df



