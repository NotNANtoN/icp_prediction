import json
from argparse import Namespace


def read_args(parser, args_dict):
    if args_dict is None:
        args = parser.parse_args()
    else:
        default_args_dict = get_argparse_defaults(parser)
        default_args_dict.update(args_dict)
        args = Namespace(**default_args_dict)

    # If we are training from a specific hyperparam set:
    if hasattr(args, 'from_config') and args.from_config:
        args = update_args_from_config(args)
    return args


def get_argparse_defaults(parser):
    """Returns the default parameters of an argparser as a dict without the required arguments
    :type parser: argparse.ArgumentParser
    """
    defaults = {}
    for action in parser._actions:
        if not action.required and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def update_args_from_config(args):
    """If we want to start a training using a specific set of hyperparameters.
    """
    config = json.load(args.from_config)
    new_args = dict(args)
    new_args.update(config["args"])
    return Namespace(**new_args)
