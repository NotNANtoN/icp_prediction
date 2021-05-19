import sys

from src.preprocess_utils.iterative_imputer import iterative_imputation


# Handling of missings

def drop_empty_target(df):
    df = df[~(df["POD"].isna())]  # & df["POCD"].isna())]
    assert df["POD"].isna().to_numpy().sum() == 0
    return df


def _mean_imputation(df):
    df_means = df.mean(axis=0)
    df_mean_filled = df.copy()
    df_mean_filled["PreCI_dichotomous_T0"].fillna(df["PreCI_dichotomous_T0"].mode()[0], inplace=True)
    df_mean_filled = df_mean_filled.fillna(df_means)
    return df_mean_filled


def _median_imputation(df):
    df_median_filled = df.fillna(df.median())
    return df_median_filled


def _minus_one_imputation(df):
    df_minuses = df.fillna(-1)
    return df_minuses


def _all_basic(df):
    filled_dict = {"mean_filled": _mean_imputation(df),
                   "median_filled": _median_imputation(df),
                   "minus_filled": _minus_one_imputation(df)}
    return filled_dict


def fill_missings(df, fill_method, estimator):
    print("Filling missing values...")
    df_no_targets = df.drop(columns=["POD", "POCD"])
    print("Num cols in missing: ", len(df.columns))
    for col in df_no_targets.columns:
        if df_no_targets[col].isna().sum().sum() == len(df_no_targets[col]):
            print(col, "is nan slice. Only NaNs in this columns")
    if fill_method == 'mean':
        df_imputed = _mean_imputation(df_no_targets)
    elif fill_method == 'median':
        df_imputed = _median_imputation(df_no_targets)
    elif fill_method == 'minus':
        df_imputed = _minus_one_imputation(df_no_targets)
    elif fill_method == 'all_basic':
        df_imputed = _all_basic(df_no_targets)
    elif fill_method == 'iterative':
        df_imputed = iterative_imputation(df_no_targets, estimator)
    else:
        sys.exit("Aborting - Issues with --fill_method flag: Invalid argument.")
    assert len(df) == len(df_imputed)
    assert len(df_imputed) == len(df['POD'])
    # TODO: why do I need to convert to list to not have NaNs appear in some cases???
    df_imputed['POD'] = list(df['POD'])  # .astype(int)
    df_imputed['POCD'] = list(df['POCD'].fillna(-1).astype(int))

    # print(len(df_imputed['POD']))
    # print(len(df['POD']))
    # print(list(df['POD']))
    # print(list(df_imputed['POD']))
    assert df_imputed['POD'].isna().to_numpy().sum() == 0
    assert df_imputed.isna().to_numpy().sum() == 0, df_imputed.isna().sum()
    print("Done filling values.")
    return df_imputed
