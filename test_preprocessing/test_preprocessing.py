from unittest import TestCase, main as unittest_main
import sys

sys.path.insert(0, '..')
from data_preprocessing import setup_and_start_preprocessing


class PreprocessingTestCase(TestCase):
    def test_all_no_save(self):
        """Test all preprocessing types without storing dataframes"""
        preprocess_args = {"save": False, "all": True}
        out = setup_and_start_preprocessing(preprocess_args)
        self.assertTrue(out)


if __name__ == '__main__':
    unittest_main()
