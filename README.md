# Pharmaimage

[![pipeline status](https://gitlab.com/%{project_path}/badges/%{default_branch}/pipeline.svg)](https://gitlab.com/%{project_path}/-/commits/%{default_branch})
[![coverage report](https://gitlab.com/%{project_path}/badges/%{default_branch}/coverage.svg)](https://gitlab.com/%{project_path}/-/commits/%{default_branch})

## Setting up your environment

Use `python -m pip install requirements.txt` to install the required packages.

## How to run

### Preprocessing 
First make sure to download the raw data `blutdaten.xlsx` into a folder `data` in the project root.

In `src` run `data_preprocessing.py`and set flags wrt. how you want to preprocess the
data.
Run `python data_preprocessing.py -h` to print the below info on all possible flags:

```
usage: data_preprocessing.py [-h] [--remove_outliers REMOVE_OUTLIERS]
                             [--norm_method {minmax,z}]
                             [--fill_method {mean,median,minus,all_basic,iterative}]
                             [--estimator {knn,trees,bayesridge}]
                             [--yeo | --no-yeo]
                             [--missingness_feature | --no-missingness_feature]
                             [--splits | --no-splits] [--kfold | --no-kfold]
                             [-k NUM_FOLDS] [--exp_id {1,2,3,4,5}]

Preprocess all data. All combinations of methods are allowed.

optional arguments:
  -h, --help            show this help message and exit
  --remove_outliers REMOVE_OUTLIERS
                        Enter quantile for which to remove outliers (e.g. `0.9999`). If flag is not used, no outlier removal will be performed. Default: None.
  --norm_method {minmax,z}
                        Normalization method. Default: minmax.
  --fill_method {mean,median,minus,all_basic,iterative}
                        Method for handling missings:
                                                     - `all_basic` runs all three basic methods (mean, median, minus) and stores them in separate folders
                                                     - `iterative` applies the IterativeImputer, you may change the used estimator with the `--estimator`
                                                          flag (for further parameter adjustments: `utils/iterative_imputer.py`)
                                                     Default: median.
  --estimator {knn,trees,bayesridge}
                        Estimator for IterativeImputer. `trees` = ExtraTreesRegressor. Default: bayesridge.
  --yeo                 To apply Yeo Johnson transform just use the --yeo flag (no param). Default: False.
  --no-yeo              Do not apply Yeo Johnson.
  --missingness_feature
                        Create a missingness feature (for each case, a list of 0 and 1 encoding where a feature is/was NaN). Default: True
  --no-missingness_feature
                        Do not create a missingness feature.
  --splits              Automatic creation of train and holdout sets. Default: True.
  --no-splits           No automatic creation of train and holdout sets.
  --kfold               Automatic generation of k folds. Num of folds can be adjusted with `-k` flag. Default: True.
  --no-kfold            No automatic creation of k folds.
  -k NUM_FOLDS, --num_folds NUM_FOLDS
                        Number of folds for k-fold as integer. Default: 5.
  --exp_id {1,2,3,4,5}  Choose experiment: 
                                                     1 - Use yeo, outlier removal (quantile 0.9999), z-stand., miss. feature, iter. imputer
                                                     2 - Use no yeo, outlier removal (quantile 0.9999), min-max norm., miss. feature, iter. imputer
                                                     3 - Use yeo, outlier removal (quantile 0.9999), z-stand., miss. feature, median fill
                                                     4 - Use yeo, outlier removal (quantile 0.9999), z-stand., iter. imputer
                                                     5 - Use yeo, z-stand., miss. feature, iter. imputer
```

As you can see, there are a couple of options for creating predefined setting configurations via the `--exp_id` flag.

For the IterativeImputer, you may enter the estimator via the `--estimator` flag. Other parameters, however, will have to be changed within `utils/iterative_imputer.py`.

The script will automatically store the preprocessed data as well as train/test 
splits of it (if not flagged otherwise).


### How to train models

Run the training script:
```
python train.py -df non_transformed_minmax/median_filled -m rf
```
Arguments (required):\
`-df`: data folder, either 'data_mean_filled' or 'data_minus_filled'\
`-m`: model, see -h for supported models\
The script has model-specific arguments, see `-h` for more information.

### How to tune hyperparameters

Run the tuning script:
```
python tune.py -df mock -m torch
```

Arguments (required):\
`-df`: data folder, either 'data_mean_filled' or 'data_minus_filled'\
`-m`: model for which to tune hyperparameters, see -h for supported models\
`-nt`: number of trials to run the optimization for\
`-nf`: number of folds to use per trial\ 
See `-h` for more information.
