import os
import json

import hydra
import omegaconf

from train_utils import default_args
from tune_utils import tune, store_study_results
from data_utils import get_seq_list, make_split
    

@hydra.main(config_path="configs", config_name="tune")
def main(cfg):
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
    # load data
    seq_list = get_seq_list(args["minutes"], False, "ICP_Vital", args["features"])
    # make dev/test split to test tuning
    dev_data, test_data, dev_idcs, test_idcs = make_split(seq_list, test_size=0.2)
    # tune
    study = tune(dev_data, test_data, args, model_args, verbose=False, n_trials=args["n_trials"],
                 opt_flat_block_size=args["opt_flat_block_size"], opt_augs=args["opt_augs"], opt_fill_type=args["opt_fill_type"])
    # store results
    store_study_results(study, "./tune_results/", test_idcs, args)
    

if __name__ == '__main__':
    main()
