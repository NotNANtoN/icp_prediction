import argparse


def create_argparser():
    """ Returns an Argparser for commandline input for custom preprocessing settings."""
    parser = argparse.ArgumentParser(description='Preprocess all data. All combinations of methods are allowed.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--save', type=int, default=1, help='Whether to save the results or to just run it. ')
    parser.add_argument('--remove_outliers', type=float, default=0.9999,
                        help='Enter quantile for which to remove outliers (e.g. `0.9999`). If flag is not used, '
                             'no outlier removal will be performed. Default: None.')

    parser.add_argument('--norm_method', type=str, default='z', choices=['minmax', 'z'],
                        help='Normalization method. Default: minmax.')

    parser.add_argument('--fill_method', type=str, default='median',
                        choices=['mean', 'median', 'minus', 'all_basic', 'iterative'],
                        help='Method for handling missings:\n \
                            - `all_basic` runs all three basic methods (mean, median, minus) and stores them in '
                             'separate folders\n \
                            - `iterative` applies the IterativeImputer, you may change the used estimator with '
                             'the `--estimator`\n \
                                 flag (for further parameter adjustments: `utils/iterative_imputer.py`)\n \
                            Default: median.')

    parser.add_argument('--estimator', type=str, default='bayesridge', choices=['knn', 'trees', 'bayesridge'],
                        help='Estimator for IterativeImputer. `trees` = ExtraTreesRegressor. Default: bayesridge.')

    parser.add_argument('--yeo', default=1, type=int,
                        help='To apply Yeo Johnson transform just use the --yeo flag (no param). Default: False.')

    parser.add_argument('--remove_multi_outliers', default=0, type=int,
                        help='Remove multivariate outliers as detected by an Isolation Forest. Default: True')

    parser.add_argument('--exp_id', type=int, choices=[1, 2, 3, 4, 5],
                        help='Choose experiment: \n \
                            1 - Use yeo, outlier removal (quantile 0.9999), z-stand., miss. feature, iter. imputer\n \
                            2 - Use no yeo, outlier removal (quantile 0.9999), min-max norm., miss. feature,'
                             ' iterative imputer\n \
                            3 - Use yeo, outlier removal (quantile 0.9999), z-stand., miss. feature, median fill\n \
                            4 - Use yeo, outlier removal (quantile 0.9999), z-stand., iter. imputer\n \
                            5 - Use yeo, z-stand., miss. feature, iter. imputer')

    parser.add_argument('--all', action='store_true', help="Creates all possible combinations of datasets")
    parser.add_argument('--remove_precipitals', default=1, type=int,
                        help="Remove some post-operative factors such as surgery duration")
    return parser


def check_exp_id(args):
    if args.exp_id == 1:
        args.yeo = True
        args.remove_outliers = 0.9999
        args.norm_method = 'z'
        args.fill_method = 'iterative'

    elif args.exp_id == 2:
        args.yeo = False
        args.remove_outliers = 0.9999
        args.norm_method = 'minmax'
        args.fill_method = 'iterative'

    elif args.exp_id == 3:
        args.yeo = True
        args.remove_outliers = 0.9999
        args.norm_method = 'z'
        args.fill_method = 'median'

    elif args.exp_id == 4:
        args.yeo = True
        args.norm_method = 'z'
        args.fill_method = 'iterative'

    return args
