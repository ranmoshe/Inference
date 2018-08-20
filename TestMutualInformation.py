import numpy as np
import pandas as pd
import unittest

from sample_set import SampleSet

class TestMutualInformation(unittest.TestCase):

    def test_probability_calc_one_column(self):
        matrix = pd.read_csv('test_files/test_probability_calc.csv', dtype=np.str)
        samples = SampleSet(matrix)
        a_prob = samples.get_probability(['a'])
        self.assertEqual(a_prob['catA'], 0.3)
        self.assertEqual(a_prob['catB'], 0.25)
        self.assertEqual(a_prob['catC'], 0.45)

    def test_probability_calc_shuffled_columns(self):
        matrix = pd.read_csv('test_files/test_probability_calc_shuffled_columns.csv', dtype=np.str)
        samples = SampleSet(matrix)
        a_prob = samples.get_probability(['b', 'a'])
        self.assertEqual(a_prob['1,catA'], 0.2)
        self.assertEqual(a_prob['2,catA'], 0.1)

    def test_pure_set(self):
        pass

    def test_a_b_mutual_info(self):
        pass


if __name__ == '__main__':
    unittest.main()

