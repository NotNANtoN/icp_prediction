""" Test script for training a model. Uses the UCI ML Breast Cancer mock dataset."""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import mlflow
from parameterized import parameterized
from unittest import TestCase, main as unittest_main
from train import main, get_args


# Every entry in the list corresponds to one model configuration.
sklearn_configs = [("k_fold", {'model_type': 'rf', 'nf': 5, 'rf_num_est': 10, 'rf_max_depth': 1}),
                   ("random_forest", {'model_type': 'rf', 'rf_num_est': 100, 'rf_max_depth': 100}),
                   ("xgb", {'model_type': 'xgb', 'xgb_num_est': 100, 'xgb_max_depth': 3, 'xgb_lr': 0.1}),
                   ("svc", {'model_type': 'svc', 'svc_c': 1.0, 'svc_tol': 1e-3}),
                   ("mlp", {'model_type': 'mlp', 'mlp_hl': (100,), 'mlp_act': 'relu', 'mlp_solver': 'adam', 'mlp_lr': 'constant',
                            'mlp_lr_init': 0.001, 'mlp_max_iter': 200, 'mlp_tol': 1e-4, 'mlp_v': False})]

torch_configs = [("multi", {'pocd': 1}),
                 ("multi_2", {'pocd': 1, 'eval_only_pod': 1}),
                 ("batchnorm", {'use_softmax': 0, 'augm_std': 0.01, 'batch_size': 8, 'bn': 1}),
                 ("softmax", {'use_softmax': 1, 'augm_std': 0, 'dropout': 1}),
                 ("ensemble", {'ensemble_k': 3}),
                 ("ensemble_all", {'ensemble_k': 3, 'ensemble_prior': 1, 'ensemble_bootstrap': 1})]


class TrainTestCase(TestCase):

    def check_model(self, trial_args):
        """Add up some more arguments for all tests."""
        default_args = {'df': 'mock', 'split': 'train/val', 'dt': 'data', 'nf': 0, 'save': False, 'v': 0}
        args = get_args({**default_args, **trial_args})
        try:
            self.assertTrue(main(args))
        except BaseException:
            # End mlflow check_model even if test crashes such that other tests can still start
            mlflow.end_run()
            raise

    @parameterized.expand(torch_configs)
    def test_torch(self, name, args):
        """Test the pytorch model configurations."""
        default_args = {'model_type': 'torch', 'hidden_size': 2, 'batch_size': 4, 'max_eps': 2}
        default_args.update(args)
        self.check_model(default_args)

    @parameterized.expand(sklearn_configs)
    def test_sklearn(self, name, args):
        """Test the sklearn model configurations."""
        self.check_model(args)


if __name__ == '__main__':
    unittest_main()
