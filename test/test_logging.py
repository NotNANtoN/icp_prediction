import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from unittest import TestCase
from io import StringIO
from src.utils.logging import log_time
from contextlib import redirect_stdout
from argparse import Namespace


class LoggingTestCase(TestCase):
    def test_log_time(self):
        # Create a dummy function to use the logger on.
        @log_time
        def dummy_func(args):
            return f"Hello AdaLab: verbose = {args.v}"
        # Call the annotated function with -v arguments.
        ns = Namespace(**{'v': True})
        # Redirect the print statements from sdout into a variable
        with redirect_stdout(StringIO()) as f:
            result = dummy_func(ns)
            self.assertEqual(result, "Hello AdaLab: verbose = True")
            self.assertRegex(f.getvalue(), 'Starting dummy_func...')
