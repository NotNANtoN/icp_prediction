import numpy as np
import pandas as pd
from model import LitRNN, LitTransformer, LitMLP
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


# define model
def create_model(model_type, feature_names, model_args):
    if model_type == "rnn":
        model = LitRNN(feature_names, 
                       hidden_size=model_args["hidden_size"], 
                       dropout_val=model_args["dropout"], 
                       rnn_layers=model_args["rnn_layers"], 
                       rnn_type=model_args["rnn_type"],
                       lr=model_args["lr"] * 0.2,
                        )
    elif model_type == "transformer":
        model = LitTransformer(feature_names,
                               ninp=512, # embedding dimension
                               nhead=16, # 2, # num attention heads
                               nhid=1024, #the dimension of the feedforward network model in nn.TransformerEncoder
                               nlayers=8, # the number of heads in the multiheadattention models
                               dropout=0.2,
                               lr=model_args["lr"] * 0.001  #0.00005,
                               )
    elif model_type  == "mlp":
        model = LitMLP(feature_names, 
                       hidden_size=256, 
                       dropout_val=0.2, 
                       lr=model_args["lr"],)
    #model = torch.jit.script(model)
    #model = model.to_torchscript()
    return model


def create_trainer(model_args, verbose=True):
    # default logger used by trainer
    logger = None
    logger = pl.loggers.mlflow.MLFlowLogger(
        experiment_name='default', 
    )
    logger.log_hyperparams(args)
    # early stopping and model checkpoint
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=False, mode="min", check_on_train_epoch_end=False)
    #mc = ModelCheckpoint(monitor='val_loss', verbose=False, save_last=False, save_top_k=1,
    #                     save_weights_only=False, mode='min', period=None)
    callbacks = [es]
    # verbosity
    verbose_kwargs = {"progress_bar_refresh_rate": 1.0 if verbose else 0.0,
                      "weights_summary": "top" if verbose else None
                     }
    if not verbose:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        pytorch_lightning.utilities.distributed.log.setLevel(logging.ERROR)
        


    #row_log_interval=k
    #log_save_interval=k

        
    # create trainer
    trainer = pl.Trainer(
                        checkpoint_callback=False,
                        callbacks=callbacks,
                        precision=16,
                        max_steps=model_args["max_steps"],
                        gradient_clip_val=model_args["grad_clip_val"],
                        track_grad_norm=-1, # -1 to disable
                        truncated_bptt_steps=None, # None to disable, 5 was mentioned in docs
                        logger=logger,
                        gpus=1,
                        val_check_interval=50,#args["val_check_interval"],
                        auto_lr_find=False,
                        **verbose_kwargs,
                        #overfit_batches=1,
                        #limit_val_batches=0.0
    ) 
    return trainer


def train_nn(model_type, data_module, model_args, verbose=True):
    # init model and trainer
    model = create_model(model_type, data_module, model_args)
    trainer = create_trainer(model_args, verbose=verbose)
    # do actual training
    trainer.fit(model, data_module)
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
        plt.ylim(val_loss_epoch["val"].min() * 0.9, val_loss_epoch["val"].max() * 1.1)
        plt.show()
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
        from xgboost import XGBRegressor
        clf = XGBRegressor(n_estimators=model_args["n_estimators"],
                           max_depth=model_args["max_depth"],
                           min_child_weight=model_args["min_child_weight"],
                           gamma=model_args["gamma"],
                           subsample=model_args["subsample"],
                           colsample_bytree=model_args["colsample_bytree"],
                           tree_method=model_args["tree_method"],
                          )
    #tree_method, set it to hist or gpu_hist
    elif model_type == "rf":
        from xgboost import XGBRFRegressor
        clf = XGBRFRegressor(n_estimators=model_args["n_estimators"],
                             max_depth=model_args["max_depth"],
                             min_child_weight=model_args["min_child_weight"],
                             gamma=model_args["gamma"],
                             subsample=model_args["subsample"],
                             colsample_bytree=model_args["colsample_bytree"],
                             tree_method=model_args["tree_method"],)
    elif model_type == "linear":
        from sklearn.linear_model import ElasticNet
        clf = ElasticNet(alpha=model_args["alpha"], 
                         l1_ratio=model_args["l1_ratio"])
    clf.fit(inputs, targets)
    return clf, data_module


def train_model(model_type, data_modules, model_args, verbose=True):
    classical_models = ["linear", "xgb", "rf"]
    num_trains = len(data_modules)
    models = []
    trainers = []
    for data_module in data_modules:
        if model_type in classical_models:
            model, trainer = train_classical(model_type, data_module, model_args, verbose=verbose)
        else:
            model, trainer = train_nn(model_type, data_module, model_args, verbose=verbose)
        # store model
        models.append(model)
        trainers.append(trainer)
    return models, trainers