""" Test script for tuning a model. Uses the UCI ML Breast Cancer mock dataset."""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import mlflow
from parameterized import parameterized
from unittest import TestCase, main as unittest_main
from train import get_args
from src.tune_src.tune_utils import run_optimization
from tune import get_tune_args


# Every entry in the list corresponds to one tuning configuration.
tune_configs = [("torch", {'model_type': 'torch'}),
                ("rf", {'model_type': 'rf'}),
                ("parallel", {'model_type': 'rf', 'pp': 2}),
                ("xgb", {'model_type': 'xgb'}),
                ("svc", {'model_type': 'svc'}),
                ("mlp", {'model_type': 'mlp'}),
                ]


class TuneTestCase(TestCase):
    def check_tune(self, trial_args):
        """Add up some more arguments for all tests."""
        default_args = {'df': 'mock', 'split': 'train/val', 'dt': 'data', 'nf': 0, 'save': False, 'v': 0, 'nt': 2,
                        'save_tune': False}
        tune_args = get_tune_args({**default_args, **trial_args})
        args = get_args({**tune_args})
        try:
            out = run_optimization(args, args.nt)
            self.assertTrue(out)
        except BaseException:
            # End mlflow check_model even if test crashes such that other tests can still start
            mlflow.end_run()
            raise

    @parameterized.expand(tune_configs)
    def test_tune(self, name, args):
        """Test the regular tuning for all model types"""
        self.check_tune(args)


if __name__ == '__main__':
    unittest_main()
