import copy

import numpy as np
import optuna
from omegaconf import open_dict

import train
from data_preprocessing import out_dir_name
from src.tune_src.best_args_util import store_args_to_cache


class NoEvalError(Exception):
    """No evaluation criterion is set. Necessary for optimization."""
    pass


class Objective:
    """Objective for optuna hyperparameter optimization.

    Runs trials for a specified classifier model with different hyperparameter suggestions.
    """

    def __init__(self, skf_idcs, train_kwargs, tuning_ranges, df, m, features, freeze_prepro, cache_dir, metric,
                 pp):
        self.opt_over_dataset = df == "opt"
        self.use_preselected_features = features
        self.classifier_name = m  # ['torch', 'rf', 'xgb', 'svc', 'mlp', 'log']
        self.train_kwargs = train_kwargs
        self.tuning_ranges = tuning_ranges
        self.model_args = train_kwargs["model_args"]
        self.freeze_prepro = freeze_prepro
        self.chosen_df = df
        self.cache_dir = cache_dir
        self.metric = metric
        self.skf_idcs = skf_idcs
        # self.best_train_args = None
        # self.best_score = None

        # Do test training:
        print("Verbose test train run:")
        if pp:
            for train_idcs, val_idcs in self.skf_idcs:
                self.train_kwargs["v"] = 1
                score, _ = train.start_training(train_idcs=train_idcs, val_idcs=val_idcs,
                                                score_fnc_name=self.metric,
                                                **self.train_kwargs)
                break
        self.train_kwargs["v"] = 0

    def __call__(self, trial):
        # get function that realizes call for given classifier
        if not hasattr(self, self.classifier_name):
            raise NotImplementedError(
                    f"Class `{self.__class__.__name__}` does not implement `{self.classifier_name}`")
        suggest_alg_hyperparams = getattr(self, self.classifier_name)
        suggest_alg_hyperparams(trial)
        # set hyperparameter suggestions
        self.suggest_general_hyperparams(trial)
        store_args_to_cache(self.cache_dir, self.train_kwargs, trial.number)

        # run training and evaluation
        try:
            monitor = 'val_' + self.metric
            callbacks = [optuna.integration.PyTorchLightningPruningCallback(trial, monitor=monitor)]
            scores = []
            for train_idcs, val_idcs in self.skf_idcs:
                score, _ = train.start_training(train_idcs=train_idcs, val_idcs=val_idcs,
                                                callbacks=callbacks,
                                                score_fnc_name=self.metric,
                                                **self.train_kwargs)
                scores.append(score)
            scores = np.array(scores)
            mean_score = scores.mean()
        except NameError:
            raise NoEvalError("No evaluation score provided. Make sure to use a validation or test split.")
        return mean_score

    def add_cat(self, trial, key):
        if type(self.tuning_ranges[key]) in (int, str):
            return
        with open_dict(self.model_args):
            self.model_args[key] = trial.suggest_categorical(key, self.tuning_ranges[key])

    def add_int(self, trial, key):
        if type(self.tuning_ranges[key]) in (int, str):
            return
        with open_dict(self.model_args):
            self.model_args[key] = trial.suggest_int(key, *self.tuning_ranges[key])

    def add_logun(self, trial, key):
        if type(self.tuning_ranges[key]) in (int, str):
            return
        with open_dict(self.model_args):
            self.model_args[key] = trial.suggest_loguniform(key, *self.tuning_ranges[key])

    def add_float(self, trial, key):
        if type(self.tuning_ranges[key]) in (int, str):
            return
        with open_dict(self.model_args):
            self.model_args[key] = trial.suggest_float(key, *self.tuning_ranges[key])

    def tabnet(self, trial):
        """Hyperparameter suggestions for Random Forest"""
        self.add_int(trial, 'n_d')
        self.add_int(trial, 'n_a')
        self.add_int(trial, 'n_steps')
        self.add_float(trial, 'gamma')
        self.add_int(trial, 'n_independent')
        self.add_int(trial, 'n_shared')
        self.add_float(trial, 'momentum')
        self.add_float(trial, 'clip_value')
        self.add_float(trial, 'lambda_sparse')
        self.add_cat(trial, 'mask_type')
        
    def rf(self, trial):
        """Hyperparameter suggestions for Random Forest"""
        self.add_int(trial, 'n_estimators')
        self.add_int(trial, 'max_depth')

    def xgb(self, trial):
        """Hyperparameter suggestions for XGBoost"""
        self.add_int(trial, 'n_estimators')
        self.add_int(trial, 'max_depth')
        self.add_float(trial, 'learning_rate')
        self.add_float(trial, 'subsample')
        self.add_float(trial, 'lambda')
        self.add_float(trial, 'alpha')
        self.add_float(trial, 'gamma')

    def svc(self, trial):
        """Hyperparameter suggestions for Support Vector Machine"""
        self.add_float(trial, 'C')
        self.add_float(trial, 'tol')
        self.add_cat(trial, 'kernel')

    def log(self, trial):
        """Hyperparameter suggestions for Support Vector Machine"""
        self.add_float(trial, 'C')
        self.add_float(trial, 'l1_ratio')

    def mlp(self, trial):
        """Hyperparameter suggestions for the sklearn Multilayer Perceptron"""
        layers = trial.suggest_int('layers', *self.tuning_ranges.layers)
        nodes = trial.suggest_int('nodes', *self.tuning_ranges.nodes)
        layers_nodes = f'({nodes},' + f' {nodes},' * (layers - 1) + ')'
        self.model_args["hidden_layer_sizes"] = layers_nodes  # ast.literal_eval(layers_nodes)
        self.add_cat(trial, 'activation')
        self.add_cat(trial, 'solver')
        self.add_cat(trial, 'learning_rate')
        self.add_logun(trial, 'learning_rate_init')
        self.add_int(trial, 'max_iter')

    def _set_self_norm_related_args(self, trial):
        self_normalizing = trial.suggest_categorical('self_norm', [0, 1])
        if self_normalizing:
            self.model_args['act_fnc'] = "selu"
            self.model_args['w_init'] = "snn"
            self.add_float(trial, 'alpha_dropout')
        else:
            self.add_cat(trial, 'act_fnc')
            if self.model_args['act_fnc'] == "relu":
                self.model_args['w_init'] = "he"
            elif self.model_args['act_fnc'] == "tanh":
                self.model_args['w_init'] = "xavier"
            self.add_cat(trial, 'bn')
            self.add_float(trial, 'dropout')

    def torch(self, trial):
        """Hyperparameter suggestions for the PyTorch MLP"""
        # Set snn related args: weigh_init, act_function, batchnorm/dropout
        self._set_self_norm_related_args(trial)

        # Set general architecture args
        self.add_float(trial, 'augm_std')
        self.add_int(trial, 'n_layers')
        self.add_int(trial, 'hidden_size')
        self.add_logun(trial, 'lr')
        self.add_int(trial, 'batch_size')
        self.add_cat(trial, 'optimizer')
        # Set ensemble args
        self.add_int(trial, 'ensemble_k')
        if self.model_args['ensemble_k'] > 1:
            self.add_cat(trial, 'ensemble_prior')
            self.add_cat(trial, 'ensemble_bootstrap')

    def suggest_general_hyperparams(self, trial):
        if self.classifier_name not in ['mlp']:
            self.train_kwargs["use_class_weights"] = 0
            self.train_kwargs["pos_weight"] = 0
            # self.add_cat(trial, 'use_class_weights')
            # if not self.model_args['use_class_weights']:
            #    self.add_float(trial, 'pos_weight')
        # Dataset options:
        # Preprocess options:
        if self.freeze_prepro:
            # yeo_N/normalization_None/median/uni_clip_0.9999/multi_clip_N
            args_from_path = self.chosen_df.split('/')
            yeo = int(args_from_path[0].endswith('Y'))
            norm_method = args_from_path[1].split('_')[-1]
            norm_method = norm_method if norm_method != 'None' else None
            fill_method = args_from_path[2]
            remove_outliers = args_from_path[3].split('_')[-1]
            remove_multi_outliers = int(args_from_path[4].endswith('Y'))
            miss_feats = 0
        elif self.opt_over_dataset:
            yeo = trial.suggest_int('yeo', *self.tuning_ranges['yeo'])
            fill_method = trial.suggest_categorical('fill', self.tuning_ranges['fill'])
            norm_method = trial.suggest_categorical('norm', self.tuning_ranges['norm'])
            remove_outliers = trial.suggest_categorical('remove_outs', self.tuning_ranges['remove_outliers'])
            remove_multi_outliers = 0  # trial.suggest_int('remove_multi_outliers',
            #                  *self.tuning_ranges['remove_multi_outliers'])
            miss_feats = 0  # trial.suggest_int('miss_feats', *self.tuning_ranges['miss_feats'])
        else:
            raise NotImplementedError("Need to either freeze preprocessing or optimize over it, choosing another "
                                      "fixed dataset is not implemented atm")
        self.train_kwargs['df'] = out_dir_name(yeo, norm_method, fill_method, remove_outliers, remove_multi_outliers)

        self.train_kwargs['miss_feats'] = miss_feats
        #print(self.train_kwargs['df'])

        # Features selection:
        """
        if self.opt_over_features:
            self.train_kwargs['blood'] = trial.suggest_int('blood', *self.tune_range['blood'])
            self.train_kwargs['imaging'] = trial.suggest_int('imaging', *self.tune_range['imaging'])
            self.train_kwargs['clinical'] = trial.suggest_int('clinical', *self.tune_range['clinical'])
            self.train_kwargs['imaging'] = trial.suggest_int('imaging', *self.tune_range['imaging'])
            if self.train_kwargs['imaging']:
                self.train_kwargs['imaging_pca'] = 0
            else:
                self.train_kwargs['imaging_pca'] = trial.suggest_int('imaging_pca', *self.tune_range['imaging_pca'])
                if self.train_kwargs['imaging_pca']:
                    self.train_kwargs['imaging_pca'] = trial.suggest_categorical(
                            'imaging_pca_var', *self.tune_range['imaging_pca_var'])
        else:
        """
