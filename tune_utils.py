import time
import operator

import optuna
import numpy as np

from train_utils import train_model
from eval_utils import get_all_dfs
from data_utils import do_fold



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


def tune(dev_data, test_data, args, model_args, verbose=True, n_trials=50):
    study_name = "tune_test_" + str(time.time())[-7:]

    if verbose:
        print(study_name)
    # Create Objective
    objective = Objective(dev_data, test_data, args, model_args, verbose=verbose)

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

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    early_stopping = EarlyStoppingCallback(30, direction=direction)
        
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, callbacks=[early_stopping])
    return study



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
                flat_block_size = trial.suggest_float("flat_block_size", 0, 10)
                for dm in self.data_modules:
                    dm.flat_block_size = flat_block_size
                    dm.make_flat_arrays()
        
        # model hyperparameters
        if self.model_type == "linear":
            self.add_float(trial, "alpha", [0.1, 10])
            self.add_float(trial, "l1_ratio", [0.01, 0.99])
        elif self.model_type == "xgb" or self.model_type == "rf":
            self.add_int(trial, "n_estimators", [3, 200])
            self.add_int(trial, "max_depth", [1, 12])
            self.add_float(trial, "learning_rate", [0.01, 0.5])
            self.add_float(trial, "subsample", [0.6, 1.0])
            self.add_float(trial, "min_child_weight", [1, 20.0])
            self.add_float(trial, "colsample_bytree", [0.6, 1.0])
            self.add_float(trial, "gamma", [0.01, 20.0])
        elif self.model_type in dl_models:
            self.add_float(trial, "lr", [0.00003, 0.005])
            self.add_float(trial, "grad_clip_val", [0.1, 10.0])
            self.add_int(trial, "hidden_size", [64, 1024])
            self.add_float(trial, "dropout", [0.0, 0.5])
            # need to set bs manually
            bs = trial.suggest_int("bs", 1, 7)
            for dm in self.data_modules:
                dm.batch_size = 2 ** bs 
            
            if self.model_type == "rnn":
                self.add_int(trial, "rnn_layers", [1, 5])
                self.add_cat(trial, "rnn_type", ["lstm", "gru"])
            
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
        

        import time
import operator


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


def tune(dev_data, test_data, args, model_args, verbose=True, n_trials=50, **kwargs):
    study_name = "tune_test_" + str(time.time())[-7:]

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

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    early_stopping = EarlyStoppingCallback(30, direction=direction)
        
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True, callbacks=[early_stopping])
    return study
