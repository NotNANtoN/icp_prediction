import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from model import LitRNN, LitTransformer, LitMLP, LucidTransformer, LitCLIP, LitGPT
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb

from data_utils import do_fold, SeqDataModule

def make_train_val_fold(df, cfg, folds, test_df=None):
    dms = []
    if folds > 0:
        pat_list = [pat[1] for pat in df.groupby("Pat_ID")]
        
        from data_utils import make_fold
        train_data_list, val_data_list, train_idcs_list, val_idcs_list = make_fold(pat_list, k=folds)
        
        for val_data, train_data in zip(train_data_list, val_data_list):
            train_df = pd.concat(train_data)
            train_df["split"] = "train"
            val_df = pd.concat(val_data)
            val_df["split"] = "val"
            df_list = [train_df, val_df]
            if test_df is not None:
                test_df["split"] = "test"
                df_list.append(test_df)
            df = pd.concat(df_list)
        
            dm = create_dm(df, cfg)
            dms.append(dm)
    else:
        dm = create_dm(df, cfg)
        dms.append(dm)
    return dms


def create_dm(df, cfg):
    dm = SeqDataModule(df,
                        target_name=cfg["target_name"],
                        random_starts=cfg["random_starts"], 
                        min_len=cfg["min_len"], 
                        max_len=cfg["max_len"],
                        train_noise_std=cfg["train_noise_std"], 
                        batch_size=cfg["bs"], 
                        fill_type=cfg["fill_type"], 
                        flat_block_size=cfg["flat_block_size"],
                        target_nan_quantile=cfg["target_nan_quantile"],
                        target_clip_quantile=cfg["target_clip_quantile"],
                        input_nan_quantile=cfg["input_nan_quantile"],
                        input_clip_quantile=cfg["input_clip_quantile"],
                        block_size=cfg["block_size"],
                        subsample_frac=cfg["subsample_frac"],
                        randomly_mask_aug=cfg["randomly_mask_aug"],
                        agg_meds=cfg["agg_meds"],
                        )
    dm.setup()
    return dm

def retrain(dev_data, test_data, best_params, cfg, verbose=True):
    # create splits
    data_modules = do_fold(dev_data, test_data, 
                           cfg["dbs"], 
                           cfg["random_starts"], 
                           cfg["min_len"], 
                           cfg["train_noise_std"],
                           cfg["bs"],
                           cfg["fill_type"], 
                           cfg["flat_block_size"],
                           k_fold=0, 
                           num_splits=10,
                          )
    # load best params
    for key in best_params:
        cfg[key] = best_params[key]
    # retrain
    models, trainers = train_model(cfg["model_type"], data_modules, cfg, verbose=True)
    if verbose:
        # print metrics
        df = get_all_dfs(models, trainers, cfg["model_type"], dl_type="test")
        print_all_metrics(df)
        loss = df.groupby("model_id").apply(lambda model_df: model_df["error"]).mean()
        print()
        print("Loss: ", loss)
        print("Std of loss: ", df.groupby("model_id").apply(lambda model_df: model_df["error"].std()))
    return models, trainers




# define model
def create_model(model_type, data_module, cfg):
    general_keys = ["weight_decay", "max_epochs", "use_macro_loss",
                    "use_pos_weight", "use_nan_embed", "lr", "use_huber",
                    "use_static", "freeze_nan_embed", "norm_nan_embed", "nan_embed_size"]
    general_kwargs = {key: cfg[key] for key in general_keys}
    if model_type == "rnn":
        model = LitRNN(data_module, 
                       hidden_size=cfg["hidden_size"], 
                       dropout_val=cfg["dropout"], 
                       rnn_layers=cfg["rnn_layers"], 
                       rnn_type=cfg["rnn_type"],
                       **general_kwargs
                        )
    elif model_type == "transformer" or model_type == "lucid_transformer":
        model_class = LitTransformer if model_type == "transformer" else LucidTransformer
        model = model_class(data_module,
                               ninp=cfg["emb_dim"], # embedding dimension
                               nhead=cfg["n_heads"], # 2, # num attention heads
                               nhid=cfg["hidden_size"], #the dimension of the feedforward network model in nn.TransformerEncoder
                               nlayers=cfg["n_layers"], # the number of heads in the multiheadattention models
                               dropout=cfg["dropout"],
                                **general_kwargs
                               )
    elif model_type  == "mlp":
        model = LitMLP(data_module, 
                       hidden_size=cfg["hidden_size"], 
                       dropout_val=cfg["dropout"], 
                       **general_kwargs)
    elif model_type == "clip":
        model = LitCLIP(data_module,
                        clip_name=cfg["clip_name"],
                        **general_kwargs)
    elif model_type == "gpt":
        model = LitGPT(data_module,
                       gpt_name=cfg["gpt_name"],
                       mode=cfg["mode"],
                       pretrained=cfg["pretrained"],
                       reduction_factor=cfg["reduction_factor"],
                       **general_kwargs)
    else:
        raise ValueError("Unknown model_type: " + str(model_type))
        
    #model = torch.jit.script(model)
    #model = model.to_torchscript()
    return model


def create_trainer(cfg, verbose=True, log=True):
    # default logger used by trainer
    #logger = None
    #logger = pl.loggers.mlflow.MLFlowLogger(
    #    experiment_name='default', 
    #)
    
    if log:
        wandb_logger = pl.loggers.WandbLogger(name=None, save_dir=None, offline=False, id=None,
                                              anonymous=None, version=None, project=cfg["target_name"],
                                              log_model=False, experiment=None, prefix='')
        hyperparam_dict = {**cfg}
        wandb_logger.log_hyperparams(hyperparam_dict)
    else:
        wandb_logger = None

    # log gradients and model topology
    #wandb_logger.watch(lit_model)

    
    # early stopping and model checkpoint
    #es = EarlyStopping(monitor='val_loss', patience=3, verbose=False, mode="min", check_on_train_epoch_end=False)
    #mc = ModelCheckpoint(monitor='val_loss', verbose=False, save_last=False, save_top_k=1,
    #                     save_weights_only=False, mode='min', period=None)
    callbacks = []
    # verbosity
    verbose_kwargs = {"enable_progress_bar": True if verbose else False,
                      "enable_model_summary": True if verbose else False,
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
                        max_steps=cfg["max_steps"],
                        gradient_clip_val=cfg["grad_clip_val"],
                        max_epochs=cfg["max_epochs"],
                        track_grad_norm=-1, # -1 to disable
                        logger=wandb_logger,
                        gpus=1,
                        #val_check_interval=100,#cfg["val_check_interval"],
                        auto_lr_find=cfg["auto_lr_find"],
                        **verbose_kwargs,
                        #overfit_batches=1,
                        #limit_val_batches=0.0
    ) 
    return trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_nn(model_type, data_module, cfg, verbose=True, log=True):
    # init model and trainer
    model = create_model(model_type, data_module, cfg)
    # print number of trainable parameters if verbose
    if verbose:
        print("Number of trainable parameters: ", count_parameters(model))
    trainer = create_trainer(cfg, verbose=verbose, log=log)
    # track gradients etc
    if verbose:
        trainer.logger.watch(model)
        
    if cfg["auto_lr_find"]:
        #trainer.tune(model, data_module.train_dataloader())
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, data_module.train_dataloader(), min_lr=1e-6, max_lr=1e-2)

        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        #model.hparams.lr = new_lr
        model.lr = new_lr

        
    # do actual training
    trainer.fit(model, data_module)
    wandb.finish(quiet=True)
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
    """
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
    """
    return model, trainer


def train_classical(model_type, data_module, cfg, verbose=True):
    # prep data to have one time step as input
    inputs = data_module.train_dataloader().dataset.flat_inputs
    targets = data_module.train_dataloader().dataset.flat_targets
    # bring into right shape and avoid NaN targets
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
        clf = XGBClass(n_estimators=cfg["n_estimators"],
                           max_depth=cfg["max_depth"],
                           min_child_weight=cfg["min_child_weight"],
                           gamma=cfg["gamma"],
                           subsample=cfg["subsample"],
                           colsample_bytree=cfg["colsample_bytree"],
                           tree_method=cfg["tree_method"],
                           eval_metric="logloss",
                           seed=cfg["seed"],
                          )
    #tree_method, set it to hist or gpu_hist
    elif model_type == "rf":
        from xgboost import XGBRFRegressor, XGBRFClassifier
        if data_module.regression:
            XGBClass = XGBRFRegressor
        else:
            XGBClass = XGBRFClassifier
        clf = XGBClass(n_estimators=cfg["n_estimators"],
                             max_depth=cfg["max_depth"],
                             min_child_weight=cfg["min_child_weight"],
                             gamma=cfg["gamma"],
                             subsample=cfg["subsample"],
                             colsample_bytree=cfg["colsample_bytree"],
                             tree_method=cfg["tree_method"],)
    elif model_type == "linear":
        from sklearn.linear_model import ElasticNet, LogisticRegression
        if data_module.regression:
             clf = ElasticNet(alpha=cfg["alpha"], 
                              l1_ratio=cfg["l1_ratio"])
        else:        
            clf = LogisticRegression(penalty="elasticnet", 
                                     solver="saga",
                                     C=cfg["C"], 
                                     l1_ratio=cfg["l1_ratio"],
                                     max_iter=cfg["max_iter"])
    clf.fit(inputs, targets)
    return clf, data_module


def train_model(model_type, data_modules, cfg, verbose=True, log=True):
    classical_models = ["linear", "xgb", "rf"]
    models = []
    trainers = []
    for data_module in data_modules:
        if model_type in classical_models:
            model, trainer = train_classical(model_type, data_module, cfg, verbose=verbose)
        else:
            model, trainer = train_nn(model_type, data_module, cfg, verbose=verbose, log=log)
        # store model
        models.append(model)
        trainers.append(trainer)
    return models, trainers
