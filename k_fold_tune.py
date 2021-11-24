import os
import json
import tracemalloc


import hydra
import omegaconf

from train_utils import default_args
from tune_utils import tune, store_study_results, make_study_name
from data_utils import get_seq_list, make_split, make_fold


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
   

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
    
    snapshot = tracemalloc.take_snapshot()
    import numpy as np
    np.save("snapshot", snapshot)
    top_stats = snapshot.statistics('lineno')
    np.save("top_stats", top_stats)
    
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
        
    
    display_top(snapshot)
    
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
