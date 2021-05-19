import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

sys.path.append('..')
from src.torch_src.torch_sklearn_wrapper import add_model_specific_args
from train import _init_model
from train import get_parser as train_parser
from src.utils.load_data_utils import get_data
from src.utils.plot_utils import plot_rfe_scores
from src.utils.args import read_args


def get_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--use_best_args', default=0, type=int,
                        help='Use hyperparameters that worked best in previous tuning. Set to int which corresponds'
                             'to the rank. E.g. 1 corresponds to the best, 2 to the first runner up. | RFE')
    parser.add_argument('--study', default=None, type=str,
                        help='Provide name of the optuna study as given in the respective subfolder name'
                             '(within optuna_studies) to reuse best model args. RFE')
    parser.add_argument('--rfe_ratio', default=0.1, type=float,
                        help='How many features to remove each step. Can be the number, or a percentage. | RFE')
    parser.add_argument('--rfe_ksplits', default=5, type=int, help='number of splits for the cross validation. | RFE')
    parser.add_argument('--rfe_njobs', default=-1, type=int, help='How many parallel processes to start. | RFE')
    parser.add_argument('--rfe_scoring', default='roc_auc',
                        help='A string or a scorer function with signature ``scorer(estimator, X, y)`` | RFE')
    parser.add_argument('-v', action='store_true', help='Use flag to store rfe artifacts')
    return parser


def get_args_for_best_model(model_rank, study, parser):
    # Load args for model of rank model_rank (1 = best) and update main_args
    study_folder = os.path.join("optuna_studies", study)
    for file in os.listdir(study_folder):
        if file.endswith(f"rank{model_rank}.pkl"):
            best_model_args = joblib.load(os.path.join(study_folder, file))
            updated_args = read_args(parser, vars(best_model_args))
            print(f"RFE: Using settings of study {study} to instantiate classifier.")
            return updated_args

    print(f"ABORTING - There are no args for a model of rank {model_rank}. Can't launch RFE.")
    return None


def _get_base_path():
    if os.getcwd()[-12:] == "/pharmaimage":
        base_path = "src"
    elif os.getcwd() == "src":
        base_path = ""
    else:
        base_path = "../"
    return base_path


def main(args, study=None):
    # Populate training and testing data structures
    args.split = 'dev/test'  # set the split to dev/test
    x_devs, y_devs, x_tests, y_tests, n_features, feature_names, class_weights = get_data(args)
    x_dev, y_dev, x_test, y_test = pd.DataFrame(x_devs[0], columns=feature_names), y_devs[0], x_tests[0], y_tests[0]

    # Initialise the model
    model = _init_model(args, feature_names, class_weights, [])

    # Run the Recursive Feature Elimination
    rfecv = RFECV(estimator=model, step=args.rfe_ratio, cv=StratifiedKFold(args.rfe_ksplits), min_features_to_select=1,
                  scoring=args.rfe_scoring, verbose=args.v, n_jobs=args.rfe_njobs)
    rfecv.fit(x_devs[0], y_devs[0])

    selected_cols = [x_dev.columns[i] for i, indicator in enumerate(rfecv.support_) if indicator]
    score = rfecv.grid_scores_[-1]
    print("grid scores", rfecv.grid_scores_)
    print(f"Original number of features is {x_devs[0].shape[1]}")
    print(f"Optimal number of features is {rfecv.n_features_} with a score of {score}")
    print(f"Optimal features are : {selected_cols}")

    # Save hyperparams and optimal feature set to disk.
    if args.v:
        run_id = datetime.now()
        configuration = {"time": run_id, "score": score, "args": vars(args), "features": selected_cols}
        with open('../rfe_artefacts/configurations.json', 'rb+') as fp:
            # remove the last two chars (']' and '\n')
            fp.seek(-2, os.SEEK_END)
            fp.truncate()
            # Add new entry line
            fp.write(bytes(",\n", 'utf-8'))
            # dump configuration object
            fp.write(json.dumps(configuration, default=str).encode('utf-8'))
            # add closing chars to the file buffer again
            fp.write(bytes("\n]", 'utf-8'))

    # Store selected features as list
    if study is not None:
        # Store in study folder if applicable
        joblib.dump(selected_cols, os.path.join("optuna_studies", study, "RFE_features.pkl"))
    else:
        # Else store in artifacts folder
        joblib.dump(selected_cols, os.path.join("rfe_artefacts", "RFE_features.pkl"))

    # Plot the performance in relation to the number of features.
    if args.write_plots:
        plot_rfe_scores(rfecv.grid_scores_, f"plots/rfe_features_{run_id}.png")


if __name__ == '__main__':
    # Fetch all the different parsers
    rfe_parser = get_parser()
    parent_parser = train_parser()
    main_parser = ArgumentParser(parents=[rfe_parser, parent_parser])
    main_parser = add_model_specific_args(main_parser)
    params = main_parser.parse_args()

    study_name = params.study
    if params.use_best_args > 0:
        params = get_args_for_best_model(params.use_best_args, study_name, main_parser)
    if params != None:
        main(params, study=study_name)
