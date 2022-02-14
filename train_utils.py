import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model import LitRNN, LitTransformer, LitMLP, LucidTransformer, LitCLIP, LitGPT
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from data_utils import make_split, do_fold


def default_args():
    # hyperparams
    args = {}
    # data args
    args["minutes"] = 60
    args["norm_method"] = None # z, None
    all_dbs = ["eICU", "UKE", "MIMIC"]
    args["dbs"] = ["UKE"]  #"all" # ["eICU", "UKE", "MIMIC"], all
    args["k_fold"] = 0
    args["num_splits"] = 3
    args["flat_block_size"] = 0
    args["model_type"] = "xgb"
    args["features"] = None

    model_args = {}
    # dataloader args
    model_args["train_noise_std"] = 0.01
    model_args["random_starts"] = True
    model_args["min_len"] = 20
    # preprocess args
    model_args["fill_type"] = "pat_ema" # "pat_mean", "mean", "pat_ema" "pat_ema_mask"
    # training args
    model_args["max_steps"] = 100
    model_args["val_check_interval"] = 100  # check validation performance every N steps
    model_args["grad_clip_val"] = 1.0
    #epochs = 10
    model_args["max_epochs"] = 5
    model_args["weight_decay"] = 0.1
    model_args["lr"] = 0.0001
    model_args["bs"] = 8
    # model args
    model_args["hidden_size"] = 512
    model_args["dropout"] = 0.1
    model_args["rnn_layers"] = 1
    model_args["rnn_type"] = "gru"
    # transfomrer
    model_args["n_layers"] = 3
    model_args["n_heads"] = 4
    model_args["emb_dim"] = 256
    #linear
    model_args["alpha"] = 1
    model_args["l1_ratio"] = 0.5
    # xgb+rf
    model_args["n_estimators"] = 50
    model_args["max_depth"] = 6 
    model_args["min_child_weight"] = 1 # 1-inf
    model_args["gamma"] = 0.0 # 0-inf
    model_args["subsample"] = 1.0 # 0.0-1.0
    model_args["colsample_bytree"] = 1.0 # 0.-1.0
    model_args["tree_method"] = "gpu_hist" # hist, gpu_hist
    

    args["seed"] = 2
    seed = pl.utilities.seed.seed_everything(seed=args["seed"], workers=False)
    return args, model_args


def retrain(dev_data, test_data, best_params, args, model_args, verbose=True):
    # create splits
    data_modules = do_fold(dev_data, test_data, 
                           args["dbs"], 
                           model_args["random_starts"], 
                           model_args["min_len"], 
                           model_args["train_noise_std"],
                           model_args["bs"],
                           model_args["fill_type"], 
                           args["flat_block_size"],
                           k_fold=0, 
                           num_splits=10,
                          )
    # load best params
    for key in best_params:
        model_args[key] = best_params[key]
    # retrain
    models, trainers = train_model(args["model_type"], data_modules, args, model_args, verbose=True)
    if verbose:
        # print metrics
        df = get_all_dfs(models, trainers, args["model_type"], dl_type="test")
        print_all_metrics(df)
        loss = df.groupby("model_id").apply(lambda model_df: model_df["error"]).mean()
        print()
        print("Loss: ", loss)
        print("Std of loss: ", df.groupby("model_id").apply(lambda model_df: model_df["error"].std()))
    return models, trainers




# define model
def create_model(model_type, feature_names, model_args):
    general_keys = ["weight_decay", "max_epochs", "use_macro_loss",
                    "use_pos_weight", "use_nan_embed", "lr"]
    general_kwargs = {key: model_args[key] for key in general_keys}
    if model_type == "rnn":
        model = LitRNN(feature_names, 
                       hidden_size=model_args["hidden_size"], 
                       dropout_val=model_args["dropout"], 
                       rnn_layers=model_args["rnn_layers"], 
                       rnn_type=model_args["rnn_type"],
                       **general_kwargs
                        )
    elif model_type == "transformer" or model_type == "lucid_transformer":
        model_class = LitTransformer if model_type == "transformer" else LucidTransformer
        model = model_class(feature_names,
                               ninp=model_args["emb_dim"], # embedding dimension
                               nhead=model_args["n_heads"], # 2, # num attention heads
                               nhid=model_args["hidden_size"], #the dimension of the feedforward network model in nn.TransformerEncoder
                               nlayers=model_args["n_layers"], # the number of heads in the multiheadattention models
                               dropout=model_args["dropout"],
                                **general_kwargs
                               )
    elif model_type  == "mlp":
        model = LitMLP(feature_names, 
                       hidden_size=model_args["hidden_size"], 
                       dropout_val=model_args["dropout"], 
                       **general_kwargs)
    elif model_type == "clip":
        model = LitCLIP(feature_names,
                        clip_name=model_args["clip_name"],
                        **general_kwargs)
    elif model_type == "gpt":
        model = LitGPT(feature_names,
                       gpt_name=model_args["gpt_name"],
                       **general_kwargs)
        
    #model = torch.jit.script(model)
    #model = model.to_torchscript()
    return model


def create_trainer(args, model_args, verbose=True):
    # default logger used by trainer
    #logger = None
    #logger = pl.loggers.mlflow.MLFlowLogger(
    #    experiment_name='default', 
    #)
    
    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = pl.loggers.WandbLogger(name=None, save_dir=None, offline=False, id=None, anonymous=None, version=None, project=args["target_name"],  log_model=False, experiment=None, prefix='')
    hyperparam_dict = {**args, **model_args}
    wandb_logger.log_hyperparams(hyperparam_dict)

    # log gradients and model topology
    #wandb_logger.watch(lit_model)

    
    # early stopping and model checkpoint
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=False, mode="min", check_on_train_epoch_end=False)
    #mc = ModelCheckpoint(monitor='val_loss', verbose=False, save_last=False, save_top_k=1,
    #                     save_weights_only=False, mode='min', period=None)
    callbacks = []
    # verbosity
    verbose_kwargs = {"enable_progress_bar": True if verbose else False,
                      "weights_summary": "top" if verbose else None
                     }
    if not verbose:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        pl.utilities.distributed.log.setLevel(logging.ERROR)
        
    #row_log_interval=k
    #log_save_interval=k
        
    # create trainer
    trainer = pl.Trainer(
                        enable_checkpointing=False,
                        callbacks=callbacks,
                        precision=16,
                        max_steps=model_args["max_steps"],
                        gradient_clip_val=model_args["grad_clip_val"],
                        max_epochs=model_args["max_epochs"],
                        track_grad_norm=-1, # -1 to disable
                        logger=wandb_logger,
                        gpus=1,
                        #val_check_interval=100,#args["val_check_interval"],
                        auto_lr_find=False,
                        **verbose_kwargs,
                        #overfit_batches=1,
                        #limit_val_batches=0.0
    ) 
    return trainer


def train_nn(args, model_type, data_module, model_args, verbose=True):
    # init model and trainer
    model = create_model(model_type, data_module, model_args)
    trainer = create_trainer(args, model_args, verbose=verbose)
    # track gradients etc
    trainer.logger.watch(model)
    # do actual training
    trainer.fit(model, data_module)
    #trainer.logger.unwatch(model)
    # load best checkpoint based on early stopping - not anymore. Disabled because it just slows down training by 5x
    #callbacks = trainer.callbacks
    #mc = [x for x in trainer.callbacks if isinstance(x, ModelCheckpoint)][0]
    #best_path = mc.best_model_path
    #model = model.load_from_checkpoint(best_path, data_module=model.data_module)
    # delete all other checkpoints
    #path_to_folder = os.path.join(*best_path.split("/")[:-1])
    #checkpoints_without_best = [cp for cp in os.listdir(path_to_folder) if cp not in best_path]
    #for cp in checkpoints_without_best:
    #    path_to_delete = os.path.join(path_to_folder, cp)
    #    os.unlink(path_to_delete)
    # plot loss curves
    if verbose:
        try:
            #print("Best model path: ", mc.best_model_path)
            # show loss plot
            logger = trainer.logger

            metrics_path = f"./mlruns/1/{logger.run_id}/metrics"
            #with open(os.path.join(metrics_path, "val_loss_step"), "r")
            train_loss_step = pd.read_csv(os.path.join(metrics_path, "train_loss"), delimiter=" ", header=None, names=["time", "val", "step"])
            #pd.read_csv(os.path.join(metrics_path, "train_loss_step"), delimiter=" ", header=None, names=["time", "val", "step"])
            val_loss_epoch = pd.read_csv(os.path.join(metrics_path, "val_loss_epoch"), delimiter=" ", header=None, names=["time", "val", "step"])
            sns.lineplot(x=val_loss_epoch["step"], y=val_loss_epoch["val"], label="val")
            sns.lineplot(x=train_loss_step["step"], y=train_loss_step["val"], label="train")
            #plt.ylim(val_loss_epoch["val"].min() * 0.9, val_loss_epoch["val"].max() * 1.1)
            plt.show()
        except Exception:
            print("WARNING:", Exception, "happened")
            pass

    return model, trainer


def train_classical(model_type, data_module, model_args, verbose=True):
    # prep data to have one time step as input
    inputs = data_module.train_dataloader().dataset.flat_inputs
    targets = data_module.train_dataloader().dataset.flat_targets
    # bring into right shape
    mask = ~np.isnan(targets)
    inputs = inputs[mask]
    targets = targets[mask]
    if verbose:
        print("Input, target shape: ", inputs.shape, targets.shape)
    # train
    if model_type == "xgb":
        from xgboost import XGBRegressor, XGBClassifier
        if data_module.regression:
            XGBClass = XGBRegressor
        else:
            XGBClass = XGBClassifier
        clf = XGBClass(n_estimators=model_args["n_estimators"],
                           max_depth=model_args["max_depth"],
                           min_child_weight=model_args["min_child_weight"],
                           gamma=model_args["gamma"],
                           subsample=model_args["subsample"],
                           colsample_bytree=model_args["colsample_bytree"],
                           tree_method=model_args["tree_method"],
                           eval_metric="logloss",
                          )
    #tree_method, set it to hist or gpu_hist
    elif model_type == "rf":
        from xgboost import XGBRFRegressor, XGBRFClassifier
        if data_module.regression:
            XGBClass = XGBRFRegressor
        else:
            XGBClass = XGBRFClassifier
        clf = XGBClass(n_estimators=model_args["n_estimators"],
                             max_depth=model_args["max_depth"],
                             min_child_weight=model_args["min_child_weight"],
                             gamma=model_args["gamma"],
                             subsample=model_args["subsample"],
                             colsample_bytree=model_args["colsample_bytree"],
                             tree_method=model_args["tree_method"],)
    elif model_type == "linear":
        from sklearn.linear_model import ElasticNet, LogisticRegression
        if data_module.regression:
             clf = ElasticNet(alpha=model_args["alpha"], 
                             l1_ratio=model_args["l1_ratio"])
        else:        
            clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                                          C=model_args["alpha"], 
                                                         l1_ratio=model_args["l1_ratio"])
    clf.fit(inputs, targets)
    return clf, data_module


def train_model(model_type, data_modules, args, model_args, verbose=True):
    classical_models = ["linear", "xgb", "rf"]
    num_trains = len(data_modules)
    models = []
    trainers = []
    for data_module in data_modules:
        if model_type in classical_models:
            model, trainer = train_classical(model_type, data_module, model_args, verbose=verbose)
        else:
            model, trainer = train_nn(args, model_type, data_module, model_args, verbose=verbose)
        # store model
        models.append(model)
        trainers.append(trainer)
    return models, trainers
