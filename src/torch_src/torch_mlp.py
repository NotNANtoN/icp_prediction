import torch
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from src.torch_src.torch_utils import to_tensor, weights_init


def get_act_fnc(act_fnc_str: str):
    if act_fnc_str == 'relu':
        act_fnc = torch.nn.ReLU
    elif act_fnc_str == 'tanh':
        act_fnc = torch.nn.Tanh
    elif act_fnc_str == 'selu':
        act_fnc = torch.nn.SELU
    elif act_fnc_str == 'lrelu':
        def act_fnc():
            return torch.nn.LeakyReLU(0.2, True)
    else:
        raise NotImplementedError(f"Invalid act function: {act_fnc_str}")
    return act_fnc


# TODO: enable different streams for blood vals, static vals and (maybe) missingness features
class MLPClassifier(LightningModule):
    def __init__(self, n_features, out_classes, class_weights, augm_std=0.01, lr=.0003,
                 use_softmax=True, act_fnc='relu', n_layers=2, hidden_size=128, dropout=0.2,
                 alpha_dropout=0, bn=1, w_init='he', optimizer='adam', v=0):
        super().__init__()
        self.v = v
        self.optimizer = optimizer
        self.use_softmax = use_softmax
        self.step = 0  # count of epochs trained so far
        self.learning_rate = lr
        self.augm_std = augm_std
        self.out_classes = out_classes
        # Define loss function
        self.pod_weight = 1
        self.pocd_weight = 0
        if use_softmax:
            self.loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
            out_classes *= 2
        else:
            pos_weight = None
            if class_weights is not None:
                pos_weight = class_weights[1]
            self.loss_fcn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        # Get act fnc:
        self.act_fnc = get_act_fnc(act_fnc)
        # Create network layers
        layers = []
        last_layer_size = n_features
        for _ in range(n_layers):
            layers.append(torch.nn.Linear(last_layer_size, hidden_size))
            if bn:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            if alpha_dropout > 0.0:
                layers.append(torch.nn.AlphaDropout(alpha_dropout))
            layers.append(self.act_fnc())
            last_layer_size = hidden_size

        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(hidden_size, out_classes)

        # Init weights:
        self.apply(lambda layer: weights_init(layer, w_init))

    def calc_loss(self, x, y, y_pred=None):
        if y_pred is None:
            y_pred = self(x)
        if self.use_softmax:
            y = y.long().flatten(0)
            y_pred = y_pred.view(-1, 2)
        loss = self.loss_fcn(y_pred, y)
        if self.use_softmax:
            loss = loss.view(-1, self.out_classes).squeeze()
        if self.out_classes == 2:
            pod_loss = loss[:, 0].mean() * self.pod_weight
            pocd_loss = loss[:, 1].mean() * self.pocd_weight
            loss = pod_loss + pocd_loss
        else:
            loss = loss.mean()
        return loss

    def forward(self, x):
        """Produces raw logits (no sigmoid applied yet)"""
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def _augment_data(self, x, y):
        if self.augm_std:
            x += torch.normal(0, self.augm_std, x.shape).to(self.device)
        return x, y

    def _find_threshold(self, x, y, y_pred=None):
        if self.use_softmax:
            # If softmax is used the threshold is not even used but it is set here for didactic purposes. As the larger
            # class with the larger logit of the two softmaxed classes is predicted, this implicitly assumes a threshold
            # of 0.5 applied to the softmaxed logits.
            self.threshold = 0.5
        else:
            # If we only have one output node we need to determine where to set a threshold to binarize the prediction.
            # At the moment we search for the threshold that maximizes the difference between
            # true positive and false positive rate
            if y_pred is None:
                y_pred = self.predict_proba(x)
            y_pred = y_pred.flatten().cpu()
            y = y.flatten().long().cpu()
            # TODO: calc sensitivity + specificity and use as criterion
            fpr, tpr, thresholds = roc_curve(y, y_pred)
            diff = tpr - fpr
            best_diff, threshold = max(zip(diff, thresholds), key=lambda pair: pair[0])
            self.threshold = float(threshold)

    def _binarize_pred(self, pred):
        return (pred > self.threshold).long().squeeze()

    def predict(self, x):
        """Returns a binary prediction for each class based off x"""
        y_pred_prob = self.predict_proba(x)
        y_pred = self._binarize_pred(y_pred_prob)
        return y_pred

    def _transform_outputs(self, y_pred_logits):
        """Applies the sigmoid function and outputs only the prediction for the 1 label if softmax is used.

         Do not use before loss function."""
        if self.use_softmax:
            y_pred_logits = y_pred_logits.view(-1, 2)
            # Only output the softmaxed logit for prediction of the "1" case
            y_pred_logits = torch.softmax(y_pred_logits, dim=1)[:, 1]
            y_pred_logits = y_pred_logits.view(-1, self.out_classes).squeeze()
            return y_pred_logits
        else:
            return torch.sigmoid(y_pred_logits).squeeze(-1)

    def predict_proba(self, x):
        """Applies sigmoid to prediction"""
        x = to_tensor(x).to(self.device)
        y_pred_logits = self(x)
        y_pred = self._transform_outputs(y_pred_logits)
        return y_pred

    def _get_ypred(self, x, y, y_pred, proba=False):
        if y_pred is None:
            if not torch.is_tensor(y):
                y = torch.from_numpy(y)
            if proba:
                y_pred = self.predict_proba(x)
            else:
                y_pred = self.predict(x)
        return y_pred, y

    def score(self, x, y, y_pred=None):
        """Calculates the accuracy of the classifier given x and y."""
        y_pred, y = self._get_ypred(x, y, y_pred)

        y = y.long().flatten().cpu()
        y_pred = y_pred.flatten().cpu()
        accuracy = (y_pred == y).float().mean()
        return accuracy

    def auc(self, y, y_pred):
        y = y.long().flatten().detach().cpu()
        y_pred = y_pred.flatten().detach().cpu()
        auc_score = roc_auc_score(y, y_pred)
        return torch.tensor(auc_score)

    def prauc(self, y, y_pred):
        y = y.long().flatten().detach().cpu()
        y_pred = y_pred.flatten().detach().cpu()
        prauc_score = average_precision_score(y, y_pred)
        return torch.tensor(prauc_score)

    def get_uncertainty_preds(self, x):
        x = to_tensor(x)
        all_preds_logits, _ = self(x, agg="none")
        all_preds = torch.stack([self._transform_outputs(pred) for pred in all_preds_logits])
        # Take mean and std:
        mu, std = all_preds.mean(dim=0), all_preds.std(dim=0)
        y_pred = self._binarize_pred(mu)
        return y_pred, mu, std, all_preds

    # PyTorch lightning methods:
    def get_progress_bar_dict(self):
        """ Hacky method to remove v_num from progress bar"""
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = self._augment_data(x, y)
        y_pred = self(x)
        loss = self.calc_loss(x, y, y_pred=y_pred)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred_raw = self(x)
        # First find threshold:
        y_pred_logit = self._transform_outputs(y_pred_raw)
        self._find_threshold(x, y, y_pred_logit)
        # Calc metrics:
        val_loss = self.calc_loss(x, y, y_pred=y_pred_raw)
        y_pred = self._binarize_pred(y_pred_logit)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': val_loss, 'val_y': y, 'val_y_pred_logit': y_pred_logit, 'val_y_pred': y_pred}

    def validation_epoch_end(self, outputs):
        outputs = outputs[0]
        # Aggregate metrics:
        val_y = outputs['val_y']  # torch.cat([x['val_y'] for x in outputs])
        val_y_pred = outputs['val_y_pred']  # torch.cat([x['val_y_pred'] for x in outputs])
        val_y_pred_logit = outputs['val_y_pred_logit']
        val_acc = self.score(None, val_y, y_pred=val_y_pred)
        val_auc = self.auc(val_y, val_y_pred_logit)
        val_prauc = self.prauc(val_y, val_y_pred_logit)
        self.log('val_acc', val_acc)
        self.log('val_rocauc', val_auc)
        self.log('val_prauc', val_prauc)

    def configure_optimizers(self):
        # Get optim
        if self.optimizer == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        elif self.optimizer == 'adamw':
            optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise Exception('Only SGD and Adam are supported optimizers.')
        # Get LR scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min',
                                                                    factor=0.1, patience=3,
                                                                    verbose=self.v,
                                                                    threshold=0.0001, threshold_mode='rel',
                                                                    cooldown=0,
                                                                    min_lr=0, eps=1e-08),
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            # 'monitor': 'val_checkpoint_on'
            'monitor': 'val_loss'
        }

        return [optim], [scheduler]
