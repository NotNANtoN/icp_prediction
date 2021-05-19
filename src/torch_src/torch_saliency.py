import torch
from captum.attr import NoiseTunnel, Saliency

from src.utils.importance_plot import create_saliency_plot


class SaliencyHandler:
    def __init__(self, models, target_idx, saliency_std, saliency_n_samples):
        """models can either be a list of models (k-fold) or a single model, target_idx is the target index for which
        to calculate the saliency, args are the command line arguments"""
        self.models = models
        self.target_idx = target_idx
        self.std = saliency_std
        self.n_samples = saliency_n_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def calc_saliency(self, x, smoothgrad_model):
        """Calculates the gradient of the trained network towards an input augmented by gaussian noise 'n_samples'
         times. All input gradients are then averaged to get a mean input feature saliency map, indicating how much the
         POD prediction is sensible to each input feature. A positive value means that a small increase of the value of
         this input feature would lead to an increase of the POD logit and vice versa with a negative value. """
        x = x.to(self.device)
        # Add requires grad for saliency:
        x.requires_grad = True
        attribution = smoothgrad_model.attribute(x, nt_type="smoothgrad", n_samples=self.n_samples,
                                                 stdevs=self.std, abs=False,
                                                 target=self.target_idx).cpu()
        # Normalize attribution:
        attribution /= attribution.sum()
        return attribution

    def calc_all_saliencies(self, x, y, model):
        """Calculates the saliencies for all samples in the input given a model and returns a stacked tensor
         of all saliencies"""
        model.to(self.device)
        smoothgrad_model = NoiseTunnel(Saliency(model))
        val_loader = model.create_loader(x, y, batch_size=1)
        all_saliencies = []
        for data, target in val_loader:
            saliency = self.calc_saliency(data, smoothgrad_model)
            all_saliencies.append(saliency)
        return torch.stack(all_saliencies)

    def create_plots_folds(self, x, y, feature_names, nf):
        """Calculates the mean saliency per feature over all patients and displays the ranking in a bar chart"""
        # TODO: take calc_model_saliency calculations out of this function by pulling them apart to have the option to
        #  just calculate saliencies or create plots
        if nf:
            all_model_sals = []
            for model_fold, x_fold, y_fold in zip(self.models, x, y):
                saliency_fold = self.calc_model_saliency(x_fold, y_fold, model_fold)
                all_model_sals.append(saliency_fold)
            saliencies = torch.stack(all_model_sals)
            saliency = torch.mean(saliencies, dim=0)
            std_saliency = torch.std(saliencies, dim=0)
        else:
            saliency = self.calc_model_saliency(x[0], y[0], self.models[0])
            std_saliency = None
        # Create plot:
        create_saliency_plot(saliency, std_saliency, feature_names)

    def calc_model_saliency(self, x, y, model):
        # Calculate mean saliencies over all samples:
        saliencies_all_samples = self.calc_all_saliencies(x, y, model)
        mean_saliencies = saliencies_all_samples.mean(dim=0).flatten()
        # Normalize to [0,1] across features
        normalized_saliencies = (mean_saliencies - min(mean_saliencies)) / (max(mean_saliencies) - min(mean_saliencies))
        # Set saliency in SklearnLightning wrapper for later storage:
        model.feature_importances_ = normalized_saliencies
        return normalized_saliencies

    def create_importance_plot(self, feature_saliency, feature_names):
        create_saliency_plot(feature_saliency, None, feature_names)
