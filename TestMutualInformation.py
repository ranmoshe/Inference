import numpy as np
import pandas as pd
import unittest

from sample_set import SampleSet

class TestMutualInformation(unittest.TestCase):

    def test_prob_calc_one_column(self):
        matrix = pd.read_csv('test_files/test_probability_calc.csv', dtype=np.str)
        samples = SampleSet(matrix)
        a_prob = samples.prob(['a'])
        self.assertEqual(float(a_prob.query('a=="catA"').joint_prob), 0.3)
        self.assertEqual(float(a_prob.query('a=="catB"').joint_prob), 0.25)
        self.assertEqual(float(a_prob.query('a=="catC"').joint_prob), 0.45)

    def test_probability_calc_one_column(self):
        matrix = pd.read_csv('test_files/test_probability_calc.csv', dtype=np.str)
        samples = SampleSet(matrix)
        a_prob = samples.probability(['a'])
        self.assertEqual(a_prob['catA'], 0.3)
        self.assertEqual(a_prob['catB'], 0.25)
        self.assertEqual(a_prob['catC'], 0.45)

    def test_probability_calc_shuffled_columns(self):
        matrix = pd.read_csv('test_files/test_probability_calc_shuffled_columns.csv', dtype=np.str)
        samples = SampleSet(matrix)
        a_prob = samples.probability(['b', 'a'])
        self.assertEqual(a_prob['1,catA'], 0.2)
        self.assertEqual(a_prob['2,catA'], 0.1)

    def test_entropy(self):
        matrix = pd.read_csv('test_files/test_entropy.csv', dtype=np.str)
        samples = SampleSet(matrix)
        self.assertEqual(samples.entropy(['c']), 0)
        self.assertEqual(samples.entropy(['a']), 1)
        self.assertEqual(samples.entropy(['c','a']), 1)

    def test_a_b_mutual_info(self):
        matrix = pd.read_csv('test_files/test_mutual_information.csv', dtype=np.str)
        samples = SampleSet(matrix)
        self.assertEqual(samples.mutual_information(['a'], ['b']), 1)
        


if __name__ == '__main__':
    unittest.main()

