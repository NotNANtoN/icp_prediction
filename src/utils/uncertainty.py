import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t

from src.torch_src.ensemble import Ensemble
from src.torch_src.torch_sklearn_wrapper import SklearnLightning
from src.utils.plot_utils import violin_plot, boxplot


def _uncertainty_debug_hist(y, mean_std, y_pred_binary, y_pred_means, y_pred_stds, y_preds_raw, clf):
    # Plot histogram of standard deviations:
    plot_folder = os.path.join("..", "debug_plots")
    os.makedirs(plot_folder, exist_ok=True)
    file_name = os.path.join(plot_folder, "std_hist.pdf")
    _plot_std_hist(y_pred_stds, mean_std, file_name)

    # Plot histograms of the distribution for the prediction of the ensemble for single patients:
    for idx in range(5):
        file_name = os.path.join(plot_folder, "distr_pred" + str(idx) + ".pdf")
        _plot_ensemble_distr_single_patients(clf, y[idx], y_pred_means[idx], y_pred_stds[idx], y_pred_binary[idx],
                                             y_preds_raw[:, idx], file_name)

    # Plot histograms of logits for patients separated by pod and non-pod groups:
    file_name = os.path.join(plot_folder, "probs_groups_hists.pdf")
    _plot_logits_pod_vs_nonpod(clf, y, y_pred_means, file_name)


def _plot_ensemble_distr_single_patients(clf, pat_pod, y_pred_means, y_pred_stds, y_pred_binary, y_preds_raw,
                                         file_name):
    plt.figure()
    plt.title("Patient POD: " + str(pat_pod.item()) + " - Predicted: " + str(y_pred_binary.item()) +
              " Mean: " + str(round(y_pred_means.item(), 2)) +
              " Std: " + str(round(y_pred_stds.item(), 2)))
    plt.hist(y_preds_raw, bins=15)
    plt.xlim(0, 1)
    plt.vlines(clf.threshold, 0, 10)
    plt.savefig(file_name)
    plt.close()


def _plot_logits_pod_vs_nonpod(clf, y, y_pred_means, file_name):
    pod_mask = y == 1
    non_pod_mask = ~pod_mask
    pod_preds = y_pred_means[pod_mask]
    non_pod_preds = y_pred_means[non_pod_mask]
    plt.figure()
    plt.hist(non_pod_preds, label="Non-POD", bins=25, alpha=1)
    plt.hist(pod_preds, label="POD", bins=25, alpha=0.8)
    plt.vlines(clf.threshold, 0, 10)
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(file_name)
    plt.close()


def _plot_std_hist(y_pred_stds, mean_std, file_name):
    plt.figure()
    plt.hist(y_pred_stds, bins=20)
    plt.vlines(mean_std, 0, 10)
    plt.savefig(file_name)
    plt.close()


def _std_filter(y, y_pred_binary, y_pred_means, y_pred_stds, clf, num_preds, verbose=False):
    # Accuracy for predictions where too high std preds are filtered out:
    #  cert_mask = y_pred_stds < mean_std
    alpha = 0.025
    # Calculate t-value and confident interval size
    t_value = t.ppf(1 - alpha, df=9)
    conf_int_range = y_pred_stds / math.sqrt(num_preds) * t_value
    # Predictions are uncertain which contain the threshold of the model within the confidence interval:
    uncert_mask = (y_pred_means + conf_int_range > clf.threshold) & (clf.threshold > y_pred_means - conf_int_range)
    cert_mask = ~uncert_mask
    # Recalculate the baseline distribution and the accuracy based on only the certain predictions:
    new_baseline = max(y[cert_mask].float().mean(), 1 - y[cert_mask].float().mean())
    new_acc = (y[cert_mask] == y_pred_binary[cert_mask]).float().mean()
    if verbose:
        print("Proportion included for certain preds: ", cert_mask.float().sum() / len(cert_mask))
        print("Correct frac for higher certainty: ", new_acc)
        print("POD baseline for higher certainty: ", new_baseline)
        print("Diff: ", new_acc - new_baseline)


def _plot_uncertainty_report(df):
    # Plot for each category
    timestamp = time.strftime("%Y-%model_type-%d-%H-%M-%S")
    xticks = np.arange(0.0, 0.051, 0.01)
    plot_subfolder = os.path.join("..", "plots", 'Uncertainty', timestamp)
    os.makedirs(plot_subfolder, exist_ok=True)

    for subset in ['mistakes', 'pod', 'all']:
        if subset == 'mistakes':
            title_detail = 'mistakes and correct ones'
            y_name = subset
        elif subset == 'pod':
            title_detail = 'POD and non-POD'
            y_name = subset
        else:
            title_detail = 'all data'
            y_name = None
        violin_plot(df, x_name='stds', y_name=y_name, xticks=xticks, title=f'Uncertainty measure for {title_detail}',
                    out_path=plot_subfolder, file_name=f'uncertainty_viol_{subset}')
        boxplot(df, x_name='stds', y_name=y_name, xticks=xticks, title=f'Uncertainty measure for {title_detail}',
                out_path=plot_subfolder, file_name=f'uncertainty_box_{subset}')


def _uncertainty_report(y_pred_binary, y_pred_means, y_pred_stds, y, clf, v=0):
    y = torch.tensor(y).squeeze()
    baseline = max(y.float().mean(), 1 - y.float().mean())
    acc = (y == y_pred_binary).float().mean()
    if v:
        print("Correct frac: ", acc)
        print("POD baseline: ", baseline)
        print("Diff: ", acc - baseline)

    print("shapes: ", y_pred_stds.shape, y.shape, y_pred_binary.shape)
    std_wrong = round(y_pred_stds[y != y_pred_binary].mean().item(), 4)
    std_right = round(y_pred_stds[y == y_pred_binary].mean().item(), 4)
    std_pod = round(y_pred_stds[y == 1].mean().item(), 4)
    std_non_pod = round(y_pred_stds[y == 0].mean().item(), 4)
    mean_std = round(y_pred_stds.mean().item(), 4)

    # Print standard deviations for groups:
    if v:
        print("Std for mistakes: ", std_wrong)
        print("Std for correct ones: ", std_right)
        print("Std for pod: ", std_pod)
        print("Std for non pod: ", std_non_pod)
        print("Mean std: ", mean_std)
        print()

    # Create df for plotting
    df = pd.DataFrame(y_pred_stds, columns=['stds'])
    df['mistakes'] = np.where(y == y_pred_binary, 'right', 'wrong')
    df['pod'] = np.where(y == 1, 'pod', 'non_pod')
    _plot_uncertainty_report(df)

    # Show some predictions:
    if v > 1:
        print("Target|Pred  |Std")
        for target, pred, std, count in zip(y, y_pred_means, y_pred_stds, range(10)):
            print(int(target.item()), "    |", round(pred.item(), 2), "|", round(std.item(), 3))
            if count == 10:
                break
        print()

    _std_filter(y, y_pred_binary, y_pred_means, y_pred_stds, clf, len(y_pred_binary), verbose=v)

    return mean_std


def eval_uncertainty(clf, x, y, verbose=False):
    """ Evaluates the uncertainty of a model.

    Args:
        clf:        Trained classifier model.
        x:          Input data.
        y:          Labels.
        args:       Arguments from train script.
        plot:       Whether or not to create a plot.
    """
    if isinstance(clf, SklearnLightning) and isinstance(clf.model, Ensemble):
        y_pred_binary, y_pred_means, y_pred_stds, y_preds_raw = clf.predict_uncertainty(x)
        # Print report:
        mean_std = _uncertainty_report(y_pred_binary, y_pred_means, y_pred_stds, y, clf)
        if verbose:
            # old histogram - not sure if it's still needed
            _uncertainty_debug_hist(y, mean_std, y_pred_binary, y_pred_means, y_pred_stds, y_preds_raw, clf)
    else:
        if verbose:
            print()
            print(type(clf),
                  "does not have an uncertainty prediction method, so no uncertainty evaluation is conducted.")
            print()
