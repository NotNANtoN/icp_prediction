import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    accuracy_score, average_precision_score
import pytorch_tabnet

    
from src.utils.plot_utils import plot_pr_curve, plot_roc, plot_conf_matrix, plot_roc_kfold
from src.utils.uncertainty import eval_uncertainty


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def _flatten_dict(multilevel_dict):
    # flatten dictionary (because sklearn's classification report is hierarchical)
    flat_dict = {}
    for key, subdict in multilevel_dict.items():
        if type(subdict) != dict:
            flat_dict[key] = multilevel_dict[key]
        else:
            for subkey, value in subdict.items():
                flat_dict[key + '_' + subkey] = value
    return flat_dict


def _sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    return sens, spec


def correct_target_shape(y_pred_logits, y_pred, y_true):
    eval_only_pod = False
    if isinstance(y_pred_logits, list):
        # This is the case if we have a non-torch model and predict pod and pocd
        if eval_only_pod:
            y_pred_logits = y_pred_logits[0][:, 0]
            y_pred = y_pred[:, 0]
            y_true = y_true[:, 0]
    else:
        if y_pred_logits.ndim == 2 and eval_only_pod:
            y_pred_logits = y_pred_logits[:, 0]
        else:
            # We only want the logits of the "1" class
            y_pred_logits = y_pred_logits[:, 1]
    return y_pred_logits, y_pred, y_true


def create_preds_and_reshape(clf, x, y):
    y_pred_logits = clf.predict_proba(x)
    y_pred = clf.predict(x)
    y_pred_logits, y_pred, y_true = correct_target_shape(y_pred_logits, y_pred, y)
    return y_pred_logits, y_pred, y_true


def auc_score(clf, x, y):
    y_pred_logits, _, y_true = create_preds_and_reshape(clf, x, y)
    if sum(np.isnan(y_pred_logits)):
        print("Warning: prediction contains NaNs! ")
        return 0
    elif sum(np.isinf(y_pred_logits)):
        print("Warning: prediction contains infs!")
        return 0
    else:
        return roc_auc_score(y_true, y_pred_logits)


def acc_score(clf, x, y):
    _, y_pred, y_true = create_preds_and_reshape(clf, x, y)
    return accuracy_score(y_true, y_pred)


def ap_score(clf, x, y):
    y_pred_logits, _, y_true = create_preds_and_reshape(clf, x, y)
    if sum(np.isnan(y_pred_logits)):
        print("Warning: prediction contains NaNs! ")
        return 0
    elif sum(np.isinf(y_pred_logits)):
        print("Warning: prediction contains infs!")
        return 0
    else:
        return average_precision_score(y_true, y_pred_logits)


def f1_score(clf, x, y):
    y_pred_logits, y_pred, y_true = create_preds_and_reshape(clf, x, y)
    return sklearn.metrics.f1_score(y_true, y_pred)


def apply_all_metrics(y_true, y_pred, y_pred_logits, shape_is_correct=False):
    if not shape_is_correct:
        y_pred_logits, y_pred, y_true = correct_target_shape(y_pred_logits, y_pred, y_true)
    report_dict = _flatten_dict(classification_report(y_true, y_pred, output_dict=True))
    report_dict['roc_auc'] = roc_auc_score(y_true, y_pred_logits)
    report_dict['pr_auc'] = average_precision_score(y_true, y_pred_logits)
    report_dict['sensitivity'], report_dict['specificity'] = _sensitivity_specificity(y_true, y_pred)
    return report_dict


def _get_feature_importances(clf, path, feature_names, fold='', verbose=0):
    importance_df = pd.DataFrame([], columns=feature_names)
    importance_df.loc[0, :] = clf.feature_importances_
    imp_sorted = importance_df.sort_values(0, axis=1, ascending=False)
    dest_path = os.path.join(path, f"{type(clf).__name__}_feature_importances{fold}.csv")
    if verbose > 1:
        print(f"Five most important features \n {imp_sorted[imp_sorted.columns[:5]]}")
        print(f"\n\n --> Storing feature importances at: {dest_path}.\n")
    importance_df.to_csv(dest_path)
    return importance_df


def predict_and_apply_metrics(clf, x_eval, y_eval, path, fold='', verbose=0):
    """Extracts predictions from the given classifier and calculates metrics for those."""
    eval_uncertainty(clf, x_eval, y_eval, verbose=verbose)
    y_pred_logit = clf.predict_proba(x_eval)
    y_pred = clf.predict(x_eval)
    report_dict = apply_all_metrics(y_eval, y_pred, y_pred_logit)
    dest_path = os.path.join(path, f"{type(clf).__name__}_metrics{fold}.json")
    json.dump(report_dict, open(dest_path, 'w'))
    if verbose:
        for key, value in report_dict.items():
            print(f"\n{key} : {value}")
    if verbose:
        print(f"\n\n --> Storing evaluation metrics at: {dest_path}.\n")
    return report_dict


def eval_clf(clfs, x_eval, y_eval, path, feature_names, verbose=0):
    if len(clfs) == 1:
        predict_and_apply_metrics(clfs[0], x_eval[0], y_eval[0], path, verbose=verbose)
        if hasattr(clfs[0], "feature_importances_"):
            _ = _get_feature_importances(clfs[0], path, feature_names, verbose=verbose)
        plot_all(clfs[0], x_eval[0], y_eval[0], path, verbose=verbose)
    else:
        # if kfold was used, evaluate each model separately and then compute and store mean metrics
        sub_path = os.path.join(path, "per_fold")
        os.makedirs(sub_path, exist_ok=True)

        # if using test set, duplicate to get list
        if not isinstance(x_eval, list):
            x_eval, y_eval = [x_eval] * len(clfs), [y_eval] * len(clfs)
        elif (len(x_eval) != len(clfs)) and len(x_eval) == 1:
            x_eval, y_eval = x_eval * len(clfs), y_eval * len(clfs)

        metrics_per_fold = None
        importances_per_fold = pd.DataFrame(columns=feature_names)

        for f in range(len(clfs)):
            # Compute all metrics (except feature importances)
            results_dict = predict_and_apply_metrics(clfs[f], x_eval[f], y_eval[f], sub_path, f, verbose=verbose)
            # Create df for collecting all results after the first results_dict has been created
            # because that is when we know which metrics are used
            if metrics_per_fold is None:
                metrics_per_fold = pd.DataFrame(columns=results_dict.keys())
            # Append to dataframe
            metrics_per_fold.loc[f, :] = list(results_dict.values())

            # Get and save feature importances:
            if hasattr(clfs[0], "feature_importances_"):
                importance_df = _get_feature_importances(clfs[f], sub_path, feature_names, f, verbose=verbose)
                importances_per_fold = importances_per_fold.append(importance_df.loc[0, :], ignore_index=True)

            # Create evaluation plots
            plot_all(clfs[f], x_eval[f], y_eval[f], sub_path, f, verbose=verbose)

        # Store aggregated metrics:
        _agg_and_store_df(metrics_per_fold, "metrics", path, type(clfs[0]))
        if importances_per_fold is not None:
            _agg_and_store_df(importances_per_fold, "importances", path, type(clfs[0]), sort=True)
        if verbose:
            print(f"\n\n --> Storing cumulative evaluation results for {len(clfs)} folds at: {path}.\n")


def _agg_and_store_df(df, name, path, clf_type, sort=False):
    df_means = df.mean()
    df_stds = df.std()
    dest_path = os.path.join(path, f"{clf_type.__name__}_cum_{name}.csv")
    merged = pd.concat([df_means, df_stds], axis=1)
    merged.columns = ["mean", "sd"]
    if sort:
        merged = merged.sort_values("mean", axis=0, ascending=False)
    merged.to_csv(dest_path)
    print(merged)


def plot_all(clf, x_eval, y_eval, path, fold='', verbose=0):
    """Creates all necessary plots based on a classifier. The classifier must be a subclass of the ClassifierMixin class
    from sklearn.base."""
    if isinstance(clf, pytorch_tabnet.tab_model.TabNetClassifier):
        return
    # Plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 16))
    plot_pr_curve(clf, x_eval, y_eval, ax=ax1)
    plot_conf_matrix(clf, x_eval, y_eval, ax=ax2)
    plot_roc(clf, x_eval, y_eval, ax=ax3)
    plot_roc_kfold([clf], x_eval, y_eval, ax=ax4)
    plt.tight_layout()
    # Save the figures to disk
    dest_path = os.path.join(path, f"{type(clf).__name__}_plots{fold}.png")
    fig.savefig(dest_path)
    plt.close()
    if verbose > 1:
        print(f"\n\n --> Stored plots at: {dest_path}.\n")
    return fig
