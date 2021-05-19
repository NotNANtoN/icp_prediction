import math

import torch


def to_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    return x


def weights_init(m, w_init):
    if not isinstance(m, torch.nn.Linear):
        return

    if w_init == 'snn':
        init_fnc = _init_weights_snn
        init_fnc(m)
    else:
        if w_init == 'he':
            init_fnc = torch.nn.init.kaiming_normal_
        elif w_init == 'xavier':
            init_fnc = torch.nn.init.xavier_normal_
        else:
            raise NotImplementedError("Unknown init function: " + str(w_init))

        init_fnc(m.weight)
        torch.nn.init.constant_(m.bias, 0)


def _init_weights_snn(layer):
    torch.nn.init.normal_(layer.weight, std=1 / math.sqrt(layer.out_features))
    if layer.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(layer.bias, -bound, bound)
