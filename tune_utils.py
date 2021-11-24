import operator
from datetime import datetime
import os
import json

import matplotlib.pyplot as plt
import optuna
import numpy as np

from train_utils import train_model
from eval_utils import get_all_dfs
from data_utils import do_fold


def store_study_results(study, path, test_idcs, args):
    name = study.study_name
    path = os.path.join(path, name)
    
    os.makedirs(path, exist_ok=True)
    # dump test idcs
    np.save(os.path.join(path, "test_idcs.npy"), test_idcs)
    # dump params
    best_params = study.best_params
    with open(os.path.join(path, "best_params.json"), "w+") as f:
        json.dump(best_params, f)
    # dump args
    with open(os.path.join(path, "args.json"), "w+") as f:
        json.dump(args, f)
    # make plots
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(path, "opt_history.png"))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(path, "param_importances_value.png"))
    optuna.visualization.matplotlib.plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration")
    plt.savefig(os.path.join(path, "param_importances_duration.png"))
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(os.path.join(path, "slice.png"))


def calc_metric(model_df):
    return model_df.groupby("ids").apply(lambda pat: pat["error"].mean()).mean()
    

class Objective:
    def __init__(self, dev_data, test_data, args, model_args, verbose=True, opt_flat_block_size=False, opt_augs=False, opt_fill_type=False):
        self.dev_data = dev_data
        self.test_data = test_data
        self.args = args
        self.model_args = model_args
        self.model_type = args["model_type"]
        self.verbose = verbose
        self.opt_flat_block_size = opt_flat_block_size
        self.opt_augs = opt_augs
        self.opt_fill_type = opt_fill_type
        
        # prepare training data
        # make train datasets
        self.data_modules = do_fold(dev_data, test_data, 
                                    args["dbs"], 
                                    model_args["random_starts"], 
                                    model_args["min_len"], 
                                    model_args["train_noise_std"],
                                    model_args["bs"],
                                    model_args["fill_type"], 
                                    args["flat_block_size"],
                                    k_fold=args["k_fold"], 
                                    num_splits=args["num_splits"],
                                   )

    def __call__(self, trial):
        # get function that realizes call for given classifier
        self.suggest_hyperparams(trial)
        
        # train model N times
        models, trainers = train_model(self.model_type, self.data_modules, self.model_args, verbose=False)

        # calc metrics over N trainings
        df = get_all_dfs(models, trainers, self.model_type, dl_type="val")

        # get avg metric
        #metric = df.groupby("model_id").apply(lambda model_df: model_df.groupby("ids")["error"].mean()).mean()
        metric = df.groupby("model_id").apply(lambda model_df: calc_metric(model_df))
        if self.verbose:
            print(metric)
        metric = metric.mean() + metric.std()

        return metric
    
    def suggest_hyperparams(self, trial):
        dl_models = ["rnn", "mlp", "transformer"]
        
        # data and general hyperparameters
        if self.opt_fill_type:
            fill_type = trial.suggest_categorical("fill_type", ["pat_mean", "mean", "pat_ema" "pat_ema_mask"])
            for dm in self.data_modules:
                dm.fill(dm.median, fill_type)
                dm.norm(dm.mean, dm.std)
                if not self.opt_flat_block_size:
                    dm.make_flat_arrays()
        
        if self.model_type in dl_models:
            if self.opt_augs:
                train_noise_std = trial.suggest_float("train_noise_std", 0.01, 0.2)
                random_starts = trial.suggest_int("random_starts", 0, 1)          
                for dm in self.data_modules:
                    dm.train_noise_std = train_noise_std
                    dm.random_starts = random_starts
                if random_starts:
                    min_len = trial.suggest_int("min_len", 1, 50)
                    for dm in self.data_modules:
                        dm.min_len = min_len
        else:
            if self.opt_flat_block_size:
                flat_block_size = trial.suggest_int("flat_block_size", 0, 10)
                for dm in self.data_modules:
                    dm.flat_block_size = flat_block_size
                    dm.make_flat_arrays()
        
        # model hyperparameters
        if self.model_type == "linear":
            self.add_float(trial, "alpha", [0.1, 10])
            self.add_float(trial, "l1_ratio", [0.01, 0.99])
        elif self.model_type == "xgb" or self.model_type == "rf":
            self.add_int(trial, "n_estimators", [3, 250])
            self.add_int(trial, "max_depth", [1, 16])
            self.add_float(trial, "learning_rate", [0.01, 0.5])
            self.add_float(trial, "subsample", [0.6, 1.0])
            self.add_float(trial, "min_child_weight", [1, 20.0])
            self.add_float(trial, "colsample_bytree", [0.6, 1.0])
            self.add_float(trial, "gamma", [0.01, 20.0])
        elif self.model_type in dl_models:
            self.add_float(trial, "grad_clip_val", [0.1, 10.0])
            self.add_int(trial, "hidden_size", [64, 1024])
            self.add_float(trial, "dropout", [0.0, 0.5])
            # need to set bs manually
            bs = trial.suggest_int("bs", 1, 5)
            for dm in self.data_modules:
                dm.batch_size = 2 ** bs 
            
            if self.model_type == "rnn":
                self.add_int(trial, "rnn_layers", [1, 10])
                self.add_cat(trial, "rnn_type", ["lstm", "gru"])
                
            if self.model_type == "transformer":
                self.add_float(trial, "lr", [0.00003, 0.1])
                self.add_int(trial, "n_layers", [1, 3])
                self.model_args["n_heads"] = 2 ** trial.suggest_int("n_heads", 0, 2)
                #self.add_int(trial, "n_heads", [1, 16])
                self.model_args["emb_dim"] = 2 ** trial.suggest_int("emb_dim", 3, 8)
                #self.add_int(trial, "emb_dim", [32, 1024])
            else:
                # allow higher lr for transformer
                self.add_float(trial, "lr", [0.00003, 0.005])

        else:
            pass

    def add_cat(self, trial, key, range_):
        self.model_args[key] = trial.suggest_categorical(key, range_)

    def add_int(self, trial, key, range_):
        self.model_args[key] = trial.suggest_int(key, *range_)

    def add_logun(self, trial, key, range_):
        self.model_args[key] = trial.suggest_loguniform(key, *range_)

    def add_float(self, trial, key, range_):
        self.model_args[key] = trial.suggest_float(key, *range_)


class EarlyStoppingCallback:
    """Early stopping callback for Optuna to stop tuning."""
    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0
        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1
        if self._iter >= self.early_stopping_rounds:
            study.stop()


def make_study_name(args):
    now = datetime.now().strftime("%m_%d_%H_%M_%S_")
    study_name = f'{now}_{args["model_type"]}_{"_".join(args["dbs"])}_{args["n_trials"]}_{args["minutes"]}'
    
    if args["opt_flat_block_size"]:
        study_name += "_optFlatBlockSize"
    if args["opt_augs"]:
        study_name += "_optAugs"
    if args["opt_fill_type"]:
        study_name += "_opt_fill_type"
    
    return study_name
    
    
def tune(dev_data, test_data, args, model_args, verbose=True, n_trials=50, **kwargs):
    study_name = make_study_name(args)

    if verbose:
        print(study_name)
    # Create Objective
    objective = Objective(dev_data, test_data, args, model_args, verbose=verbose, **kwargs)

    # Create sampler
    sampler = optuna.samplers.TPESampler()
    # Create pruner
    pruner = None#create_pruner(pruner_name="median")
    # Get database:
    storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                         engine_kwargs={"connect_args": {"timeout": 30}})
    direction = "minimize"
    study = optuna.create_study(direction=direction, 
                                study_name=study_name,
                                pruner=pruner, 
                                sampler=sampler, 
                                storage=storage)
    # store test idcs in study

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    early_stopping = EarlyStoppingCallback(30, direction=direction)
        
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, callbacks=[early_stopping])
    return study
