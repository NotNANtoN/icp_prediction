import argparse
import os
import pickle
import sys

sys.path.insert(0, '..')

from src.utils.load_data_utils import load_data
from src.utils.metrics import eval_clf


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model given a path on some data")
    parser.add_argument("-p", required=True, type=str, help="Name of the model subfolder")
    parser.add_argument("-d", required=True, type=str, help="Name of the data subfolder")
    parser.add_argument("--mode", choices=["blood", "clinical", "imaging"], default="all",
                        help="Specifies the type of data that should be evaluated on")
    # parser.add_argument("--split", choices=["val", "test", "all", "dev"], default="test",
    #                    help="Specifies the data split that should be evaluated")
    parser.add_argument("--target", choices=["POD", "POCD"], default="POD", help="Specify which target to evaluate for")
    parsed_args = parser.parse_args()
    return parsed_args


def _load_models(path):
    all_files = os.listdir(path)
    pkled_files = map(lambda y: os.path.join(path, y), filter(lambda x: ".pkl" in x, all_files))

    # unpickle all files within folder
    file_dict = {}
    for pkl in pkled_files:
        infile = open(pkl, 'rb')
        file_dict[pkl.split('/')[-1]] = pickle.load(infile)
        infile.close()

    # get hyperparameter settings of the training
    hyperparams = file_dict['model_hyperparameters.pkl']
    print(f"\n\nWas trained with the following hyperparams:\n\n\t{hyperparams}")

    # if kfold was used, extract all models and write to list
    if hyperparams.nf == 0:
        models = [file_dict['model.pkl']]
        print(f"\nUnpickled the following model:\n\n\t{models[0]}")
    else:
        num_folds = hyperparams.nf
        models = [file_dict[f'model{i}.pkl'] for i in range(num_folds)]
        print(f"\nUnpickled {num_folds} versions of the following model:\n\n\t{models[0]}")
    return models, hyperparams


def _get_relevant_data(data_path, mode, target, hyperparams):
    if os.getcwd()[-12:] == "/pharmaimage":
        base_path = "data/"
    else:
        base_path = "../data/"
    path = base_path + "/"
    # Get data
    # load data as specified by user
    hyperparams.split = 'dev/test'
    _, _, x_eval, y_eval, _, feature_names, _ = load_data(data_path, hyperparams)
    targets = y_eval
    inputs = x_eval
    return inputs, targets, feature_names


def evaluate_model(models, inputs, targets, output_path, feature_names, verbose=0):
    eval_clf(models, inputs, targets, output_path, feature_names, verbose=verbose)


if __name__ == "__main__":
    args = _parse_args()

    model_path = os.path.join("../models", args.p)
    output_path = os.path.join("../eval_results", args.p)
    os.makedirs(output_path, exist_ok=True)

    models, hyperparams = _load_models(model_path)
    inputs, targets, feature_names = _get_relevant_data(args.d, args.mode, args.target, hyperparams)

    evaluate_model(models, inputs, targets, output_path, hyperparams.nf, feature_names, hyperparams)
