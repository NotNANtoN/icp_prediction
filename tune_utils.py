import os
import datetime
import copy

import sklearn
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from train_utils import train_model
from data_utils import SeqDataModule
from eval_utils import get_all_dfs


def obj(df, cfg, opt_object, num_seeds=3, split="val", dm=None):
    # put in op_df
    cfg = copy.deepcopy(cfg)
    if isinstance(opt_object, pd.DataFrame):
        opt_dict = opt_object.iloc[0].to_dict()
    else:
        opt_dict = opt_object
    cfg = merge_params_in_cfg(cfg, opt_dict)
    # calculate metrics for number of seeds
    if num_seeds is None:
        metric = train_and_eval_model(df, cfg, split=split, log=False, dm=dm)
    else:
        metrics = []
        for seed in range(num_seeds):
            cfg["seed"] = seed
            metrics.append(train_and_eval_model(df, cfg, split=split, log=False, dm=dm))
        metric = np.mean(metrics)
    return metric


def scale_hyperparameters(opt_df):
    # scale batch size exponentially
    if "bs" in opt_df:
        opt_df["bs"] = 2 ** opt_df["bs"]
    # enable nan_embed layer if we do not fill nans
    if opt_df["fill_type"] == "none":
        opt_df["use_nan_embed"] = True
    # set min len to max len
    if "max_len" in opt_df:
        opt_df["min_len"] = opt_df["max_len"]
    # convert to int
    if "flat_block_size" in opt_df:
        opt_df["flat_block_size"] =int(opt_df["flat_block_size"])
    if "n_estimators" in opt_df:
        opt_df["n_estimators"] = int(opt_df["n_estimators"])
    if "bs" in opt_df:
        opt_df["bs"] = int(opt_df["bs"])
    if "batch_size" in opt_df:
        opt_df["batch_size"] = int(opt_df["batch_size"])

    return opt_df


def train_and_eval_model(df, cfg, split, log=True, dm=None):
    #setup dm
    if dm is None:
        dm = setup_dm(df, cfg)
    # train model on datamodule
    models, trainers = train_model(cfg["model_type"], [dm], cfg, verbose=False, log=log)
    metric = eval_model(dm, models, trainers, cfg["model_type"], split)
    return metric


def setup_dm(df, cfg):
    import pytorch_lightning as pl
    pl.utilities.seed.seed_everything(seed=cfg["seed"], workers=False)
    # create datamodule with dataloaders
    dm = SeqDataModule(df, cfg["db_name"],
                       target_name=cfg["target_name"],
                       random_starts=cfg["random_starts"], 
                       min_len=cfg["min_len"], 
                       max_len=cfg["max_len"],
                       train_noise_std=cfg["train_noise_std"], 
                       batch_size=cfg["bs"], 
                       fill_type=cfg["fill_type"], 
                       flat_block_size=cfg["flat_block_size"],
                       target_nan_quantile=cfg["target_nan_quantile"],
                       block_size=cfg["block_size"],
                       subsample_frac=cfg["subsample_frac"],
                       )
    dm.setup()
    return dm


def setup_dm_and_train(df, cfg, log=True):
    #setup dm
    dm = setup_dm(df, cfg)
    # train model on datamodule
    models, trainers = train_model(cfg["model_type"], [dm], cfg, verbose=False, log=log)
    return dm, models, trainers


def eval_model(dm, models, trainers, model_type, split):
    # make preds on val set
    pred_df = get_all_dfs(models, trainers, model_type, dm.regression, dl_type=split, dl=None, calc_new_norm_stats=False)
    
    # calc target metrics
    pred_df = pred_df.dropna(subset=["targets"])
    pred_targets = pred_df["targets"]
    preds = pred_df["preds"]
    if dm.regression:
        try:
            score = sklearn.metrics.r2_score(pred_targets, preds)
        except ValueError:
            score = -10
    else:
        try:
            score = sklearn.metrics.roc_auc_score(pred_targets, preds)
        except ValueError:
            score = 0        
    return np.array([score])


def get_val_and_test_metric(dm, models, trainers, cfg):
    val_metric = eval_model(dm, models, trainers, cfg["model_type"], "val")
    test_metric = eval_model(dm, models, trainers, cfg["model_type"], "test")
    return val_metric, test_metric


def merge_params_in_cfg(cfg, params):
    cfg = copy.deepcopy(cfg)
    # put the best hyperparameters in the config
    cfg.update(params)
    cfg = scale_hyperparameters(cfg)
    return cfg


def train_and_test(df, cfg, best_params, num_seeds=5, return_weights=False):
    # train the model with the best hyperparameters and test it on test split
    cfg = merge_params_in_cfg(cfg, best_params)

    val_scores = []
    test_scores = []
    weights = []
    for seed in tqdm(range(num_seeds), desc="Training models with best parameters", disable=num_seeds==1):
        cfg["seed"] = seed
        dm, models, trainers = setup_dm_and_train(df, cfg)
        val_score, test_score = get_val_and_test_metric(dm, models, trainers, cfg)
        
        if hasattr(models[0], "state_dict") and return_weights:
            model_weights = models[0].state_dict()  # get weights of the model
            model_weights = {k: v.cpu() for k, v in model_weights.items()}  # move to cpu
            weights.append(model_weights)
        val_scores.append(val_score)
        test_scores.append(test_score)

    if return_weights:
        return val_scores, test_scores, weights
    else:
        return val_scores, test_scores


def retrain_best_trial(study, df, cfg, folder_name):
    # get the best hyperparameters
    best_trial = study.best_trial
    best_params = best_trial.params

    val_scores, test_scores, weights = train_and_test(df, cfg, best_params, 
                                                      num_seeds=5, return_weights=True)

    # store best params and scores in a dataframe
    best_param_df = pd.DataFrame(best_params, index=[0])
    best_param_df["val_score_mean"] = np.mean(val_scores)
    best_param_df["val_score_std"] = np.std(val_scores)
    best_param_df["test_score_mean"] = np.mean(test_scores)
    best_param_df["test_score_std"] = np.std(test_scores)
    best_param_df.to_csv(f"{folder_name}/best_params.csv")

    # save cfg
    import json
    with open(f"{folder_name}/cfg.json", "w+") as f:
        json.dump(cfg, f)

    return val_scores, test_scores, weights


def get_best_params(study, num_trials=5):
    trials = study.trials
    best_trials = sorted(trials, key=lambda trial: trial.value, reverse=True)[:num_trials]
    top_params = [trial.params for trial in best_trials]
    top_vals = [trial.value for trial in best_trials]
    return top_params, top_vals


def train_multiple(param_list, df, cfg):
    # train models for the best params and get the model weights
    top_param_weights = []
    top_param_val_scores = []
    top_param_test_scores = []
    for params in tqdm(param_list, desc="Training models with best parameters", disable=len(param_list)==1):
        val_scores, test_scores, weights = train_and_test(df, cfg, params, num_seeds=1, return_weights=True)
        top_param_val_scores.append(val_scores[0])
        top_param_test_scores.append(test_scores[0])
        top_param_weights.append(weights[0])
    return top_param_val_scores, top_param_test_scores, top_param_weights


def make_optuna_foldername(cfg):
    # create folder name according to the database name, minutes, model type and date
    folder_name = f"tunings/{cfg['db_name']}_{cfg['minutes']}/{cfg['model_type']}_{cfg['opt_steps']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_study_results(study, folder_name, model_type):
    # create a dataframe with the results
    tune_result_df = pd.DataFrame(study.trials_dataframe())
    tune_result_df.to_csv(f"{folder_name}/results.csv")

    def save_plot(plot, name):
        plot.write_image(os.path.join(folder_name, name + ".png"), width=1024 * 1.5, height=800 * 1.5)

    plot = optuna.visualization.plot_slice(study)
    save_plot(plot, "slice")
    plot = optuna.visualization.plot_param_importances(study)
    save_plot(plot, "param_importances")
    plot = optuna.visualization.plot_optimization_history(study)
    save_plot(plot, "optimization_history")
    if model_type in ["rnn", "gpt", "mlp"]:
        plot = optuna.visualization.plot_contour(study, params=["lr", "bs", "weight_decay", "grad_clip_val"])
    elif model_type == "xgb":
        plot = optuna.visualization.plot_contour(study, params=["lr", "n_estimators", "max_depth", "subsample",
                                                                "colsample_bytree", "gamma", "min_child_weight",
                                                                "flat_block_size"])
    elif model_type == "linear":
        plot = optuna.visualization.plot_contour(study, params=["C", "max_iter", "flat_block_size"])
    save_plot(plot, "contour")

def suggest_deep_learning(trial: optuna.trial.Trial):
    # suggest hyperparameters for deep learning models
    rec = {'lr': trial.suggest_loguniform("lr", 1e-5, 1e-3),
            #'min_len': trial.suggest_int("min_len", 2, 128),
            #'train_noise_std': trial.suggest_float("train_noise_std", 0.001, 0.2),
            #'weight_decay': trial.suggest_float("weight_decay", 0.001, 0.4),
            #'grad_clip_val': trial.suggest_float("grad_clip_val", 0.1, 5.0),   
            #'train_noise_std': trial.suggest_int("train_noise_std", 0, 2),
            'weight_decay': trial.suggest_discrete_uniform("weight_decay", 0.0, 0.4, 0.02),
            'grad_clip_val': trial.suggest_discrete_uniform("grad_clip_val", 0, 1.5, 0.1), 
            
            #'fill_type': trial.suggest_categorical("fill_type", ["median", "none"]),
            #'max_epochs': trial.suggest_int("max_epochs", 5, 100),
            }
    return rec


def suggest_tree(trial):
    rec = {'lr': trial.suggest_loguniform("lr", 0.0005, 0.5),
            'n_estimators': trial.suggest_discrete_uniform("n_estimators", 10, 300, 10),
            'max_depth': trial.suggest_int("max_depth", 2, 10),
            'subsample': trial.suggest_discrete_uniform("subsample", 0.5, 1.0, 0.05),
            'colsample_bytree': trial.suggest_discrete_uniform("colsample_bytree", 0.3, 1.0, 0.05),
            'gamma': trial.suggest_discrete_uniform("gamma", 0.0, 3.0, 0.1),
            'min_child_weight': trial.suggest_discrete_uniform("min_child_weight", 0.0, 3.0, 0.1),
            #'fill_type': trial.suggest_categorical("fill_type", ["median", "none"]),
    }
    return rec

def suggest_logreg(trial):
    # suggest hyperparameters for sklearn logistic regression
    rec = {'C': trial.suggest_loguniform("C", 0.00005, 10.0),
           'max_iter': trial.suggest_int("max_iter", 10, 500),
           'l1_ratio': trial.suggest_discrete_uniform("l1_ratio", 0.0, 1.0, 0.05),}
    return rec

def objective_optuna(trial: optuna.Trial, df, cfg, dm=None):
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    if cfg["model_type"] in ["rnn", "gpt", "mlp"]:
        rec = suggest_deep_learning(trial)
        if cfg["model_type"] == "gpt":
            if cfg["gpt_name"] == "gpt2":
                rec["bs"] =  6 #trial.suggest_int("bs", 2, 6)
            elif cfg["gpt_name"] == "gpt2-medium":
                rec["bs"] =  4 #trial.suggest_int("bs", 2, 4)
            elif cfg["gpt_name"] == "gpt2-large":
                rec["bs"] =  3 # trial.suggest_int("bs", 2, 3)
            elif cfg["gpt_name"] == "gpt2-xl":
                rec["bs"] = 0
            elif cfg["gpt_name"] == "distilgpt2":
                rec["bs"] =  7 #trial.suggest_int("bs", 2, 7)
        else:
            rec["bs"] =  trial.suggest_int("bs", 2, 8)
        
    else:
        if cfg["model_type"] in ("xgb", "rf"):
            rec = suggest_tree(trial)
        elif cfg["model_type"] == "linear":
            rec = suggest_logreg(trial)

        if cfg["flat_block_size_range"] > 1:
            rec["flat_block_size"] =  trial.suggest_discrete_uniform("flat_block_size", 0, cfg["flat_block_size_range"], 1)
        else:
            rec["flat_block_size"] = cfg["flat_block_size"]
    score = obj(df, cfg, rec, num_seeds=cfg["num_seeds"], dm=dm)
    return score  

