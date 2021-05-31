import os
import random
import subprocess
import time

import joblib
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

from src.tune_src.best_args_util import read_best_from_cache
from src.tune_src.pruner import create_pruner
from src.tune_src.tune_objective import Objective
from src.utils.load_data_utils import get_dev_idcs_and_targets


def timestring():
    return time.strftime("%b-%d-%H:%M:%S")


def _get_loading_path(eval_path, load_path):
    return eval_path if eval_path is not None else load_path


def _load_study(study_name, root_path):
    path = os.path.join(root_path, study_name, 'study.pkl')
    print(f'Loading study from: {path}')
    study = joblib.load(path)
    return study


def _get_path(eval_path, load_path, model_name, metric, nf, pruner, nt):
    load_name = eval_path if eval_path is not None else load_path
    if load_name is None:
        timestamp = timestring()
        subfolder = f'{timestamp}_{model_name}_{metric}_{nf}_{nt}' \
                    f'{f"_{pruner}" if model_name == "torch" else ""}'
    else:
        subfolder = load_name
    subfolder = subfolder.replace(',', '_')
    return subfolder


def _store_best_trial_args(cache_dir, sorted_trial_df, path):
    best_n = 5
    trial_ids = sorted_trial_df.loc[:best_n - 1, 'number']
    best_args_list = read_best_from_cache(cache_dir, trial_ids)
    for i, best_args in enumerate(best_args_list):
        joblib.dump(best_args, os.path.join(path, f'args_trialnum{trial_ids.iloc[i]}_rank{i + 1}.pkl'))


def get_trial_args(cache_dir, trial_num):
    best_args = read_best_from_cache(cache_dir, [trial_num])[0]
    return best_args


def _store_study(study, plot_dict, root_dir, subfolder, eval_path, cache_dir):
    # Create folder for current experiment
    attrs = ('number', 'value', 'params', 'state')
    sorted_trial_df = study.trials_dataframe(attrs=attrs).sort_values('value', ascending=False).reset_index()
    path = os.path.join(root_dir, subfolder)
    os.makedirs(path, exist_ok=True)

    if not eval_path:
        print(f"\n\nOPTUNA:Storing study in: {path}")

        # Store both DataFrame with top 10 trials and study in same subfolder
        df_file = os.path.join(path, 'df.csv')
        pkl_file = os.path.join(path, 'study.pkl')

        # Store the best 10 trial parameters and outcomes
        print(f"\n\nOPTUNA: All trials\n\n\t{sorted_trial_df}")
        print("Optimized over: ", list(sorted_trial_df.columns))
        sorted_trial_df.head(10).to_csv(df_file)

        # Store the experiment to be able to resume it later
        joblib.dump(study, pkl_file)

        # Store args of n best trials
        _store_best_trial_args(cache_dir, sorted_trial_df, path)

    # Store the plots:
    for plot_name in plot_dict:
        plot = plot_dict[plot_name]
        # plot.update_layout(
        #        autosize=False,
        #        width=800,
        #        height=1024,
        #        margin=dict(
        #                l=50,
        #                r=50,
        #                b=100,
        #                t=100,
        #                pad=4
        #        ),
        #        paper_bgcolor="LightSteelBlue",
        # )
        plot.write_image(os.path.join(path, plot_name + ".pdf"))


def create_dts_dict(features):
    dts_dict = {}
    if features:
        dts_dict['features'] = features

    return dts_dict


def _create_study(train_kwargs, root_dir, subfolder_name, eval_path, load_path, metric, pruner_name,
                  save_tune, tuning_ranges, df, m, features, freeze_prepro, dev_idcs, inner_splits,
                  y_dev, pp):


    # Set cache directory
    current_time = time.strftime('%c').replace(" ", "_")
    cache_dir = f'.cache/tune_trials/{current_time}'
    # Load saved arguments, keep only nt and pp of current args
    load_name = _get_loading_path(eval_path, load_path)
    if load_name is not None:
        loaded_args = joblib.load(os.path.join(root_dir, load_name, 'args.pkl'))
        subfolder_name, metric, pruner_name, skf_idcs, train_kwargs, tuning_ranges, df, m, features, \
        freeze_prepro, cache_dir, metric = loaded_args
    # Create sampler
    sampler = optuna.samplers.TPESampler()
    # Create pruner
    pruner = create_pruner(pruner_name=pruner_name)
    # Get database:
    storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                         engine_kwargs={"connect_args": {"timeout": 30}})
    # Set data types:
    data_type_kwargs = create_dts_dict(features)

    # Set train_args:
    train_kwargs.update(data_type_kwargs)
    train_kwargs["split"] = 'train/val'

    # Create study
    if load_name is not None:
        print("Loading study: ", load_name)
        # study = _load_study(load_name, root_dir)
        study = optuna.load_study(pruner=pruner, sampler=sampler, study_name=subfolder_name,
                                  storage=storage)

    else:
        # If no dev_idcs are given, get pre-calculated dev_idcs created during preprocessing
        if dev_idcs is None:
            df_load = train_kwargs.df  #"yeo_Y/z/median/uni_clip_0.9999/multi_clip_N"
            dev_idcs, y_dev = get_dev_idcs_and_targets(df_load, 0, v=0, **data_type_kwargs)
        # Do a stratified k-fold split to perform optimization over
        skf = StratifiedKFold(n_splits=inner_splits, shuffle=True)
        skf_idcs = list(skf.split(np.zeros(len(dev_idcs)), y_dev))

        # Dump args
        path = os.path.join(root_dir, subfolder_name)
        os.makedirs(path, exist_ok=True)
        save_args = [subfolder_name, metric, pruner_name, skf_idcs, train_kwargs, tuning_ranges, df, m, features,
                     freeze_prepro, cache_dir, metric]
        joblib.dump(save_args, os.path.join(path, 'args.pkl'))

        # Create study
        if save_tune:
            study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler, study_name=subfolder_name,
                                        storage=storage)  # maximize score
        else:
            study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler, study_name=subfolder_name)

    # Create objective
    objective = Objective(skf_idcs, train_kwargs, tuning_ranges, df, m, features, freeze_prepro,
                          cache_dir, metric, pp)

    return study, objective, subfolder_name, cache_dir


def run_optimize_call(arg_tuple):
    model_type, subfolder_name, nt = arg_tuple
    subprocess.Popen([f"python3 tune.py tune_model={model_type} load_path={subfolder_name} pp=0 nt={nt}"],
                     shell=True)


def run_optimization(train_args, model_type, nt, df='mock', nf_inner=3, load_path=None,
                     eval_path=None, pruner='median', pp=0, metric='prauc', tuning_ranges=None,
                     freeze_prepro=0, features=None, dev_idcs=None, y_dev=None, save_tune=False):
    """Run the hyperparameter optimization and store parameters, metrics and study object."""
    root_dir = "optuna_studies"
    subfolder_name = _get_path(eval_path=eval_path, load_path=load_path, model_name=model_type,
                               metric=metric, nf=nf_inner, pruner=pruner, nt=nt)
    # Create dev_idcs
    # Create study and objective. Args might be changed if a study is loaded
    study, objective, subfolder_name, cache_dir = _create_study(train_args, root_dir,
                                                                subfolder_name, eval_path,
                                                                load_path, metric, pruner,
                                                                save_tune, tuning_ranges, df,
                                                                model_type, features, freeze_prepro,
                                                                dev_idcs, nf_inner, y_dev, pp)

    path = os.path.join(root_dir, subfolder_name)
    # Start parallel tuning processes
    original_num_steps = nt
    if pp > 1:
        print("Recording tuning time...")
        start_time = time.time()
    if not eval_path:
        if pp > 1:
            nt = (nt // pp) + 1
            use_mp = False
            if use_mp:
                from multiprocessing import Pool
                with Pool(pp - 1) as pool:
                    work_results = pool.map(run_optimize_call,
                                            [(model_type, subfolder_name, nt) for _ in range(pp - 1)])
            else:
                for _ in range(pp - 1):
                    time.sleep(random.random() * 3)  # take a short nap to not load stuff into the db at the same time
                    subprocess.Popen([f"python3 tune.py tune_model={model_type} load_path={subfolder_name} pp=0 nt={nt}"],
                                     shell=True)
        study.optimize(objective, n_trials=nt)

    if save_tune:
        plot_dict = {}
        # plot_dict = {"contours": optuna.visualization.plot_contour(study),
        #             "coordinates": optuna.visualization.plot_parallel_coordinate(study),
        #             "importances": optuna.visualization.plot_param_importances(study),
        #             "opt_performance": optuna.visualization.plot_optimization_history(study)}

        _store_study(study, plot_dict, root_dir, subfolder_name, eval_path, cache_dir)
        
    # wait for subprocesses to finish:
    if pp > 0:
        start_wait_time = time.time()
        while len(study.get_trials()) < original_num_steps:
            time.sleep(1)
            if time.time() - start_wait_time > 60:
                print("WARNING: Waited one minute for subprocesses to finish. Stopped the wait now...")
                break

    print(f"\n\nOPTUNA: Best trial\n\n\t{study.best_trial}")
    if pp > 0:
        print()
        print("Seconds tuning took in total: ", time.time() - start_time)
        print()
    # get best train args:
    best_train_num = study.best_trial.number
    best_train_args = get_trial_args(cache_dir, best_train_num)
    
    return study.best_value, study.best_params, study.best_trial, best_train_args
