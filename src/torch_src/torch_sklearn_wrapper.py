import logging

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.torch_src.ensemble import Ensemble
from src.torch_src.torch_mlp import MLPClassifier
from src.torch_src.torch_saliency import SaliencyHandler
from src.torch_src.torch_utils import to_tensor


class SklearnLightning(BaseEstimator, ClassifierMixin, torch.nn.Module):
    def __init__(self, feature_names, out_classes, logger, class_weights, callbacks, ensemble_k=0, batch_size=32,
                 workers=0, save=False, max_eps=50, write_plots=0, saliency_std=0.01, saliency_n_samples=50,
                 ensemble_prior=0, ensemble_bootstrap=0, v=0,
                 **model_args):
        super().__init__()
        self.v = v
        self.callbacks = callbacks
        self.is_tuning = callbacks is not None
        self.feature_names = feature_names
        self.out_classes = out_classes
        # hyperparams
        self.class_weights = class_weights
        self.ensemble_k = ensemble_k
        self.ensemble_prior = ensemble_prior
        self.ensemble_bootstrap = ensemble_bootstrap
        self.batch_size = batch_size
        self.workers = workers
        self.save = save
        self.max_eps = max_eps
        # For Sklearn compatibility:
        self._estimator_type = "classifier"
        self.classes_ = [0, 1]
        # Logging etc:
        self.logger = logger
        self.write_plots = write_plots
        # Saliency:
        self.saliency_std = saliency_std
        self.saliency_n_samples = saliency_n_samples

        self.model_args = model_args

    def _create_model(self, ensemble_k, ensemble_prior, ensemble_bootstrap, *model_args, **model_kwargs):
        if ensemble_k > 1:
            self.model_init_args = (MLPClassifier, ensemble_k, *model_args)
            self.model_init_kwargs = {'prior': ensemble_prior, 'bootstrap': ensemble_bootstrap}
            self.model_class = Ensemble
        else:
            self.model_init_args = (*model_args,)
            self.model_init_kwargs = model_kwargs
            self.model_class = MLPClassifier
        model = self.model_class(*self.model_init_args, **self.model_init_kwargs)

        return model

    def predict(self, x):
        return self.model.predict(x).cpu()

    def predict_proba(self, x):
        y_pred_one = self.predict_proba_single(x)
        y_pred_zero = 1 - y_pred_one
        return torch.stack([y_pred_zero, y_pred_one], dim=1)

    @torch.no_grad()
    def predict_proba_single(self, x):
        return self.model.predict_proba(x).cpu()

    def forward(self, x):
        x = self.model.predict_proba(x).cpu()
        if x.ndim == 1:
            x = x.unsqueeze(1)
        return x

    def score(self, x, y, y_pred=None):
        """Calculates the accuracy of the classifier given x and y."""
        return self.model.score(x, y, y_pred).double().detach()

    def create_loader(self, inputs, targets, batch_size=None, train=False):
        inputs, targets = to_tensor(inputs), to_tensor(targets)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        if batch_size is None:
            batch_size = self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.workers, shuffle=train,
                                drop_last=True if train else False)
        return dataloader

    def _create_trainer(self):
        early_stop_callback = EarlyStopping(
                monitor='val_loss',
                min_delta=0.00,
                patience=5,
                verbose=self.v,
                mode='min')

        if self.is_tuning:
            self.save = False
            # Disable GPU info logging
            logging.getLogger("lightning").setLevel(logging.ERROR)

        if not self.save:
            checkpointer = False
        else:
            checkpointer = ModelCheckpoint(filepath=None,
                                           monitor='val_auc',
                                           verbose=False,
                                           save_last=False,
                                           save_top_k=1,
                                           mode='max',
                                           period=1,
                                           prefix='')
        checkpointer = None

        trainer = Trainer(
                # overfit_batches=1,
                # train_percent_check=0.5,
                max_epochs=self.max_eps,
                gpus=1 if torch.cuda.is_available() else 0,
                # amp_level='O1',
                precision=16 if torch.cuda.is_available() else 32,
                # checkpoint_callback=checkpointer,
                progress_bar_refresh_rate=self.v,
                weights_summary="full" if self.v else None,
                # val_check_interval=1.0,
                logger=self.logger if self.v else None,
                gradient_clip_val=0,
                auto_lr_find=False,
                callbacks=self.callbacks + [early_stop_callback]
        )
        return trainer

    def fit(self, x_train, y_train, x_eval=None, y_eval=None):
        self.model = self._create_model(self.ensemble_k, self.ensemble_prior, self.ensemble_bootstrap,
                                        x_train.shape[1], self.out_classes, self.class_weights,
                                        v=self.v,
                                        **self.model_args)
        if x_eval is None and y_eval is None:
            # When fit is called via RFE, it does not provide a validation set -> so we create one.
            x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.2)
        # Convert to tensors
        x_train, y_train = to_tensor(x_train), to_tensor(y_train)
        # Get dataloader
        train_dataloader = self.create_loader(x_train, y_train, train=True)
        val_dataloader = self.create_loader(x_eval, y_eval, train=False)
        # Get trainer
        trainer = self._create_trainer()
        # Fit
        trainer.fit(self.model, train_dataloader, val_dataloader)
        # Load best model
        # self.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, 1, None)
        if self.v:
            print("Warning: Loaded model checkpoint is the last checkpoint and not the best one!")
        # Set to eval mode
        self.model.eval()
        # Calculate saliencies and feature importances and store them in the _feature_importances field
        if not self.is_tuning:
            saliency_calculator = SaliencyHandler(self, 0, self.saliency_std, self.saliency_n_samples)
            saliency = saliency_calculator.calc_model_saliency(x_eval, y_eval, self)
            if self.write_plots:
                saliency_calculator.create_importance_plot(saliency, self.feature_names)
        # Put on cpu for saving
        self.model.to("cpu")

    @torch.no_grad()
    def predict_uncertainty(self, x):
        """If the model is an ensemble, predict the uncertainty of the prediction. This is done by taking the
        standard deviation over the prediction of all ensemble members."""
        assert isinstance(self.model, Ensemble)
        # Get predictions:
        y_pred, mu, std, all_preds = self.model.get_uncertainty_preds(x)
        return y_pred, mu, std, all_preds

    @property
    def threshold(self):
        return self.model.threshold

    @threshold.setter
    def threshold(self, val):
        self.model.threshold = val
