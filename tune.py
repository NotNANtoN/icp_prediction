import os

import hydra
from hydra.experimental import compose

import train
from src.tune_src.tune_utils import run_optimization


def get_train_args_tune(cfg):
    override_dict = {'model_type': cfg.model_type, 'v': 0, 'save': 0}
    train_cfg = train.get_args(override_dict=override_dict)
    #if cfg.model_type == 'torch':
    #    train_cfg.model_args.log = 0
    return train_cfg


def get_tune_args(override_dict=None):
    # get tune args
    override_list = []
    if override_dict is not None:
        for key, value in override_dict.items():
            if value is None:
                value = ""
            override_list.append(f'{key}={value}')
    cfg = compose(config_name="tune", overrides=override_list)
    # get train_args
    train_cfg = get_train_args_tune(cfg)
    return cfg, train_cfg


def get_args_and_tune():
    cfg, train_cfg = get_tune_args()
    value, best_params = run_optimization(train_args=train_cfg, **cfg)
    return value, best_params


@hydra.main(config_path="configs", config_name="tune")
def main(cfg):
    # keep original working directory for mlflow etc
    os.chdir(hydra.utils.get_original_cwd())
    # get train_args
    train_cfg = get_train_args_tune(cfg)
    value, best_params, best_trial, best_train_args = run_optimization(train_args=train_cfg, **cfg)
    print("value: ", value, "best params: ", best_params)
    return True


if __name__ == '__main__':
    main()
