import os
import json
import tracemalloc


import hydra
import omegaconf

from icp_pred.train_utils import default_args
from icp_pred.tune_utils import tune, store_study_results, make_study_name, snapshot
from icp_pred.data_utils import get_seq_list, make_split, make_fold


@hydra.main(config_path="configs", config_name="tune")
def main(cfg):
    tracemalloc.start()
    cfg_args = cfg["args"]
    # keep original working directory for mlflow etc
    os.chdir(hydra.utils.get_original_cwd())
    # get default args
    args, model_args = default_args()
    # overwrite args
    for key in cfg_args:
        content = cfg_args[key]
        if content is not None:
            if isinstance(content, omegaconf.ListConfig):
                content = list(content)
            args[key] = content
    run_k_fold(args, model_args, return_studies=False)
    
    snapshot()
    tracemalloc.stop()


def run_k_fold(args, model_args, return_studies=True):
    # define path
    exp_name = make_study_name(args)
    # load data
    seq_list = get_seq_list(args["minutes"], False, "ICP_Vital", args["features"])
    # outer fold
    dev_data_list, test_data_list, dev_idcs, test_idcs = make_fold(seq_list, k=args["n_outer_folds"])
    # tune and store studies
    studies = []
    for i, (dev_data, test_data) in enumerate(zip(dev_data_list, test_data_list)):
        study = tune(dev_data, test_data, args, model_args, verbose=False, n_trials=args["n_trials"],
                 opt_flat_block_size=args["opt_flat_block_size"], opt_augs=args["opt_augs"], opt_fill_type=args["opt_fill_type"])
        snapshot()
        # print key tune details
        name = study.study_name
        print(name)
        print(study.best_trial)
        best_params = study.best_params
        print(best_params)
        store_study_results(study, f"./k_fold_tune/{exp_name}", test_idcs[i], args)
        if return_studies:
            studies.append(study)
        else:
            del study
    return studies
    

if __name__ == '__main__':
    main()
