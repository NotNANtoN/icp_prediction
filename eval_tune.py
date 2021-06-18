import os
import joblib

import hydra
from sklearn.model_selection import StratifiedKFold
from omegaconf import open_dict

import train
from tune import run_optimization, get_tune_args
from src.utils.load_data_utils import get_data
from src.utils.metrics import apply_all_metrics
from src.tune_src.tune_utils import timestring


@hydra.main(config_path="configs", config_name="eval_tune")
def main(cfg):
    # keep original working directory for mlflow etc
    os.chdir(hydra.utils.get_original_cwd())
    # Get path to save results
    eval_tune_folder = 'results_eval_tune'
    time_str = timestring()
    subfolder_name = f'{time_str}_{cfg.model}_{cfg.nt // 1000}k_' \
                     f'{cfg.nf_outer}_{cfg.nf_inner}'
    path = os.path.join(eval_tune_folder, subfolder_name)
    os.makedirs(path, exist_ok=True)
    # Save eval_tune hyperparams
    joblib.dump(cfg, os.path.join(path, "cfg.pkl"))
    print("Saving to ", subfolder_name)

    # Get tune cfg and overwrite some tune settings
    with open_dict(cfg):
        nf_outer = cfg.pop('nf_outer')
        cfg['tune_model'] = cfg.pop('model')
    tune_cfg, train_cfg = get_tune_args(override_dict=cfg)

    # Load data once to determine splits
    df = "yeo_N/normalization_z/median/uni_clip_0.9999/multi_clip_N"
    x_train, y_train, x_eval, y_eval, n_features, feature_names, class_weights = \
        get_data(df_name=df, split='no-split', nf=0, v=0,
                 miss_feats=0,
                 )
    x_train, y_train, x_eval, y_eval = x_train[0], y_train[0], x_eval[0], y_eval[0]
    print("Train shape. ", x_train.shape)
    skf = StratifiedKFold(n_splits=nf_outer, shuffle=True)

    store_lists = []
    for split_idx, (dev_idcs, test_idcs) in enumerate(skf.split(x_train, y_train)):
        y_dev, y_eval = y_train[dev_idcs], y_train[test_idcs]
        # Run hyperparameter tuning:

        value, hyperparams, trial, best_train_args = run_optimization(dev_idcs=dev_idcs, y_dev=y_dev, train_args=train_cfg, **tune_cfg)
        print(value, hyperparams, best_train_args)
        # Set training hyperparams to best hyperparams found during tuning
        """
        if cfg['df'] == 'opt':
            yeo = hyperparams.pop('yeo')
            norm_method = hyperparams.pop('norm')
            fill_method = hyperparams.pop('fill')
            remove_outliers = hyperparams.pop('remove_outs')
            train_cfg.df = out_dir_name(yeo, norm_method, fill_method, remove_outliers, 0)
            train_cfg.miss_feats = hyperparams.pop('miss_feats')
        """
        for key, val in best_train_args.items():
            train_cfg[key] = val
        # Based on best hyperparams, validate on test data:
        eval_score, y_pred_logits, y_pred_binary, y_true, trained_model = train.start_training(dev_idcs=dev_idcs,
                                                                                               test_idcs=test_idcs,
                                                                                               return_preds=True,
                                                                                               **train_cfg)
        print("Test set score: ", eval_score)
        y_pred_logits, y_pred_binary, y_true = y_pred_logits[0], y_pred_binary[0], y_true[0]
        score_dict = apply_all_metrics(y_true, y_pred_binary, y_pred_logits, shape_is_correct=True)

        store_list = [train_cfg, eval_score, score_dict, y_pred_logits, y_pred_binary, y_true, trained_model]
        joblib.dump(store_list, os.path.join(path, f'{split_idx}.pkl'))
        store_lists.append(store_list)


if __name__ == "__main__":
    main()
