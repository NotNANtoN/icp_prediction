import torch

from train_utils import create_model
from tune_utils import get_val_and_test_metric


@torch.no_grad()
def make_model_soup(state_dicts):
    merged = {}
    for key in state_dicts[0]:
        dtype = state_dicts[0][key].dtype
        if dtype == torch.float32 or dtype == torch.float16:
            merged[key] = torch.mean(torch.stack([state_dict[key] for state_dict in state_dicts]), axis=0)
        else:
            merged[key] = state_dicts[0][key]
    return merged


def create_soup_model(cfg, data_module, soup_weights):
    # make soup of models with different weights but same architecture
    top_param_soup = make_model_soup(soup_weights)
    # create empty model
    soup_model = create_model(cfg["model_type"], data_module, cfg)
    # put soup weights into model
    soup_model.load_state_dict(top_param_soup)
    return soup_model


def create_and_eval_soup(soup_dm, cfg, weights):
    soup_model = create_soup_model(cfg, soup_dm, weights)
    soup_val_score, soup_test_score = get_val_and_test_metric(soup_dm, [soup_model], None, cfg)
    return soup_val_score, soup_test_score