import optuna


def create_pruner(pruner_name):
    if pruner_name == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=25,
                                             n_warmup_steps=10,
                                             interval_steps=1)
    elif pruner_name == 'asha':
        pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource='auto', reduction_factor=3)
    else:
        raise NotImplementedError("Unknown pruner: " + str(pruner_name))
    return pruner
