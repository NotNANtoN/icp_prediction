import os
import shutil

import joblib


# Utils for storing and retrieving args of the best optuna trial.


def store_args_to_cache(dir_path, trial_args, trial_id):
    os.makedirs(dir_path, exist_ok=True)
    joblib.dump(trial_args, os.path.join(dir_path, f'{trial_id}.pkl'))
    return True


def read_best_from_cache(dir_path, trial_ids):
    # Read args for n best trials from cache
    trial_args_list = []
    for trial_id in trial_ids:
        trial_args_list.append(joblib.load(os.path.join(dir_path, f'{trial_id}.pkl')))
    return trial_args_list


def delete_cache():
    dir_path = '.cache/tune_trials'
    if os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    else:
        print(dir_path, "does not exist. Moving on.")
