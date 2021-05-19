import torch
import torch.nn as nn

from src.torch_src.torch_mlp import MLPClassifier


class Ensemble(MLPClassifier):
    """Creates an ensemble of models. Has support for bootstrapping the batch per model (resampling the batch with
     replacement for each ensemble member) and for the incorporation of a
     randomized prior (https://arxiv.org/abs/1806.03335)."""

    def __init__(self, ModelClass, k, *model_args, prior=False, bootstrap=False, **model_kwargs):
        super().__init__(*model_args)
        self.k = k
        self.models = nn.ModuleList([ModelClass(*model_args, **model_kwargs) for _ in range(k)])
        self.prior = prior
        self.bootstrap = bootstrap
        self.priors = self._create_priors(ModelClass, k, model_args, prior, **model_kwargs)

    @staticmethod
    def _create_priors(ModelClass, k, model_args, create_priors, **model_kwargs):
        """Creates a randomized prior network with frozen weights for every ensemble member"""
        if create_priors:
            priors = nn.ModuleList([ModelClass(*model_args, **model_kwargs) for _ in range(k)])
            # Freeze prior models:
            for prior_model in priors:
                for param in prior_model.parameters():
                    param.requires_grad = False
        else:
            priors = None
        return priors

    def _forward_single(self, model, x, idx, all_bootstrap_idcs):
        """Forward pass for single ensemble member"""
        # Bootstrap batch
        if self.bootstrap and self.training:
            bootstrap_idcs = torch.randint(low=0, high=len(x), size=(len(x),))
            model_input = x[bootstrap_idcs]
            all_bootstrap_idcs.append(bootstrap_idcs)
        else:
            model_input = x
        # Make model prediction
        pred = model(model_input)
        # Apply randomized prior
        if self.prior:
            prior_pred = self.priors[idx](model_input)
            pred += prior_pred
        return pred

    def forward(self, x, agg="mean"):
        """Make a forward pass through all members of the ensemble (and their priors, if any). Aggregate the outputs
        according to the agg method"""
        # Need to save the bootstrap indices eventual loss calculation
        all_bootstrap_idcs = []
        preds = [self._forward_single(model, x, idx, all_bootstrap_idcs) for idx, model in enumerate(self.models)]
        # Aggregate predictions
        preds = torch.stack(preds)
        if agg == "mean":
            return torch.mean(preds, dim=0)
        elif agg == "none":
            return preds, all_bootstrap_idcs

    def calc_loss(self, x, target, y_pred=None):
        """Calculates the averaged loss for all ensemble members"""
        # Get not-aggregated predictions
        preds, all_bootstrap_idcs = self(x, agg="none")

        loss = 0
        for idx in range(self.k):
            pred, model = preds[idx], self.models[idx]
            if self.bootstrap and self.training:
                bootstrap_idcs = all_bootstrap_idcs[idx]
                # Use same bootstrapping idcs on target as on input:
                model_target = target[bootstrap_idcs]
            else:
                model_target = target
            loss += model.calc_loss(None, model_target, y_pred=pred)
        loss /= self.k

        return loss
