import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor


def get_iterative_imputer(estimator_obj):
    imputer = IterativeImputer(estimator=estimator_obj, missing_values=np.nan, sample_posterior=False,
                               max_iter=10, tol=0.001, n_nearest_features=None, initial_strategy='median',
                               imputation_order='ascending', skip_complete=True, min_value=None,
                               max_value=None, verbose=0, random_state=None, add_indicator=False)
    return imputer


def iterative_imputation(df, estimator):
    estimators_dict = {'knn': KNeighborsRegressor(),
                       'trees': ExtraTreesRegressor(),
                       'bayesridge': None}

    imputer = get_iterative_imputer(estimators_dict[estimator])
    # Take out imaging data as it eats too much RAM (~1200 columns)
    imaging_names = [col for col in df.columns if "imaging" in col]
    imaging_df = df[imaging_names]
    no_imaging_df = df.drop(columns=imaging_names)
    no_imaging_df_np = no_imaging_df.to_numpy()
    print(f"Fitting imputer on shape: {no_imaging_df_np.shape}...")
    print("Imputing...")
    imputed_array = imputer.fit_transform(no_imaging_df_np)

    # Recreate df by merging the iteratively imputed df with the imaging df (filled with mean vals).
    df_imputed_data = np.concatenate([imputed_array, imaging_df.fillna(imaging_df.mean()).to_numpy()], axis=1)
    df_imputed_cols = list(no_imaging_df.columns) + list(imaging_df.columns)
    df_imputed = pd.DataFrame(data=df_imputed_data, columns=df_imputed_cols)

    return df_imputed
