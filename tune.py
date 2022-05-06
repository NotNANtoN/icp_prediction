import os
from tabnanny import verbose

import hydra
import numpy as np


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # transformers to run offline
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # change to original working directory
    os.chdir(hydra.utils.get_original_cwd())
    cfg = dict(cfg)      
    # overrides and calculated default vals
    if cfg["lr"] is None:
        model_type = cfg["model_type"]
        if model_type == "clip":
            cfg["lr"] = 0.001
        elif model_type == "gpt":
            # bs 8 and gpt2 take 9.8GB with max seq len of 512
            # bs 16 with max seq len of 256
            # bs 32 with max seq len 128 only 7.4GB, good performance and fast - 6.9 if mlp_norm
            # bs 64 with len 128 and mlp_norm = 10.9GB. 9.4GB for freeze
            cfg["lr"] = 0.00005
        else:
            cfg["lr"] = 0.0001  # 0.01 works kind of for nan_embed
                    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    # disable wandb printing
    os.environ['WANDB_SILENT'] = "true"

    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
    locals().update(cfg)
    
    import pandas as pd
    import logging
    import pytorch_lightning
    #logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    #pytorch_lightning.utilities.distributed.log.setLevel(logging.ERROR)
    
    # load df
    path = f"data/DB_{cfg['db_name']}_{cfg['minutes']}_final_df.pkl"
    df = pd.read_pickle(path)

    

    # run hebo tuning if wanted
    if cfg["tune_hebo"]:
        from tune_hebo import get_hebo_space, tune_hebo
        space = get_hebo_space(cfg["model_type"])
        tune_hebo(df, space, cfg)

    # OPTUNA
    import optuna
    import pandas as pd
    from tune_utils import objective_optuna

    
    study = optuna.create_study(direction="maximize")  # Create a new study.
    print("Setup dm...")
    from tune_utils import setup_dm
    dm = setup_dm(df, cfg)
    print("Start tuning...")
    study.optimize(lambda study: objective_optuna(study, df, cfg, dm=dm), 
                   n_trials=cfg["opt_steps"], gc_after_trial=True,
                   show_progress_bar=True,
                   )

    from tune_utils import make_optuna_foldername, save_study_results

    # create a folder to save the results
    folder_name = make_optuna_foldername(cfg)
    print("Saving in folder name: ", folder_name)
    save_study_results(study, folder_name, cfg["model_type"])

    from tune_utils import retrain_best_trial

    val_scores, test_scores, weights = retrain_best_trial(study, df, cfg, folder_name)

    # test the souping approach:
    if cfg["model_type"] in ["rnn", "gpt", "mlp"]:
        from soup_utils import create_and_eval_soup
        from tune_utils import merge_params_in_cfg, setup_dm

        soup_cfg = merge_params_in_cfg(cfg, study.best_params)
        soup_dm = setup_dm(df, soup_cfg)

        # make soup of different seeds of the best hyperparam set
        soup_val_score, soup_test_score = create_and_eval_soup(soup_dm, soup_cfg, weights)
        # save scores of soup model
        soup_scores_df = pd.DataFrame({"val_score": [soup_val_score], 
                                    "test_score": [soup_test_score],
                                    "average_ind_val_score": [np.mean(val_scores)],
                                    "average_ind_test_score": [np.mean(test_scores)],
                                    })
        soup_scores_df.to_csv(f"{folder_name}/best_param_seed_soup_scores.csv", index=False)

        # get params of some of the best optuna trials to create a soup from
        from tune_utils import get_best_params, train_multiple
        top_params, top_vals = get_best_params(study, num_trials=5)
        top_param_val_scores, top_param_test_scores, top_param_weights = train_multiple(top_params, df, cfg)
        
        # soup of some of the best hyperparameter sets
        soup_val_score, soup_test_score = create_and_eval_soup(soup_dm, soup_cfg, top_param_weights[:5])
        # save scores of soup model
        soup_scores_df = pd.DataFrame({"val_score": [soup_val_score],
                                    "test_score": [soup_test_score],
                                    "average_ind_val_score": [np.mean(top_param_val_scores)],
                                    "average_ind_test_score": [np.mean(top_param_test_scores)],
                                    "average_val_score_from_tuning": [np.mean(top_vals)],
                                    })
        soup_scores_df.to_csv(f"{folder_name}/top_param_soup_scores.csv", index=False)

    return np.mean(test_scores)


if __name__ == "__main__":
    main()