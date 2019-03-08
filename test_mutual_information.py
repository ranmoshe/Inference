import numpy as np
import pandas as pd
import random
import unittest

from sample_set import SampleSet
from utils import generate_random_a_b

class TestMutualInformation(unittest.TestCase):

    def test_probability_calc_one_column(self):
        matrix = pd.read_csv('test_files/test_probability_calc.csv', dtype=np.str)
        samples = SampleSet(matrix)
        a_prob = samples.probability(['a'])
        self.assertEqual(float(a_prob.query('a=="catA"').joint_prob), 0.3)
        self.assertEqual(float(a_prob.query('a=="catB"').joint_prob), 0.25)
        self.assertEqual(float(a_prob.query('a=="catC"').joint_prob), 0.45)

    def test_probability_calc_shuffled_columns(self):
        matrix = pd.read_csv('test_files/test_probability_calc_shuffled_columns.csv', dtype=np.str)
        samples = SampleSet(matrix)
        a_prob = samples.probability(['b', 'a'])
        self.assertEqual(float(a_prob.query('b=="1" and a=="catA"').joint_prob), 0.2)
        self.assertEqual(float(a_prob.query('b=="2" and a=="catA"').joint_prob), 0.1)

    def test_entropy(self):
        matrix = pd.read_csv('test_files/test_entropy.csv', dtype=np.str)
        samples = SampleSet(matrix)
        self.assertEqual(samples.entropy(['c']), 0)
        self.assertEqual(samples.entropy(['a']), 1)
        self.assertEqual(samples.entropy(['c','a']), 1)

    def test_a_b_mutual_info(self):
        matrix = pd.read_csv('test_files/test_mutual_information.csv', dtype=np.str)
        samples = SampleSet(matrix)
        self.assertEqual(samples.mutual_information(['a'], ['b'])['mi'], 1)
        
    def test_a_b_given_c(self):
        '''
        The test file has 200 rows where c==1, and MI(a,b)==1, 
        and 100 rows where c==2 and MI(a,b)==0
        '''
        matrix = pd.read_csv('test_files/test_conditional_mutual_information.csv', dtype=np.str)
        samples = SampleSet(matrix)
        self.assertEqual(samples.mutual_information(['a'], ['b'], ['c'])['mi'], 0.6666666666666666)

    def test_a_b_mutual_information_declining(self):
        '''
        1. Start with 99 rows, two vars, 3 categories, perfectly correlated
        2. Add 100 rows, randomly generated (different random seed for each var), see that mutual information is smaller
        3. Repeat 2, and see that mutual information is smaller yet
        '''
        # Create perfectly correlated df
        df = pd.DataFrame([], columns=list('ab'))
        for i in range(33):
            df1 = pd.DataFrame([['1','1'], ['2','2'], ['3','3']], columns=list('ab'))
            df = df.append(df1, ignore_index=True)
        sase = SampleSet(df)
        mi1 = sase.mutual_information(['a'], ['b'])['mi']
        self.assertAlmostEqual(mi1, np.log2(3))

        # Append 100 uncorrelated rows
        rows = [[str(random.randint(1,3)), str(random.randint(1,3))] for i in range(100)]
        randDf = pd.DataFrame(rows, columns=list('ab'))
        df = df.append(randDf, ignore_index=True)
        sase = SampleSet(df)
        mi2 = sase.mutual_information(['a'], ['b'])['mi']
        self.assertTrue(mi2 < mi1)
        
        # Append another 100 uncorrelated rows
        rows = [[str(random.randint(1,3)), str(random.randint(1,3))] for i in range(100)]
        randDf = pd.DataFrame(rows, columns=list('ab'))
        df = df.append(randDf, ignore_index=True)
        sase = SampleSet(df)
        mi3 = sase.mutual_information(['a'], ['b'])['mi']
        self.assertTrue(mi3 < mi2)
        
    def test_mutual_information_vs_categories(self):
        '''
        Test that mutual information is similar to log(number of categories) for perfectly correlated sets
        '''
        # test for 3 categories
        rows_3_cats = [['1', '1'], ['2', '2'], ['3', '3']]
        df = pd.DataFrame(rows_3_cats, columns=list('ab'))
        sase = SampleSet(df)
        self.assertAlmostEqual(sase.mutual_information(['a'], ['b'])['mi'], np.log2(3))

        # test for 100 categories
        rows_100_cats = [[str(i), str(i)] for i in range(100)]
        df = pd.DataFrame(rows_100_cats, columns=list('ab'))
        sase = SampleSet(df)
        self.assertAlmostEqual(sase.mutual_information(['a'], ['b'])['mi'], np.log2(100))

    def test_separate_categories(self):
        '''
        Test that if categories are not separate an exception is thrown
        '''
        rows_3_cats = [['1', '1'], ['2', '2'], ['3', '3']]
        df = pd.DataFrame(rows_3_cats, columns=list('ab'))
        sase = SampleSet(df)
        with self.assertRaises(Exception):
            sase.mutual_information(['a','b'], ['b'])

    def test_mutual_information_significance(self):
        '''
        Test that when generating A, B randomly according to P(A), P(B), the % of Type I errors matches the significance.
        '''
        errors = 0
        tests = 200
        significance = 0.95
        for i in range(tests):
            df = generate_random_a_b(100, [0.3], [0.7])
            sase = SampleSet(df)
            p_val = sase.mutual_information(['a'], ['b'])['p_val']
            if p_val > significance:
                errors += 1

        std = np.sqrt((significance*(1-significance))/tests)
        self.assertAlmostEqual(1-significance, errors/tests, delta=3*std)

    def test_mutual_information_significance_2(self):
        '''
        Test that when generating A, B randomly according to P(A), P(B), the % of Type I errors matches the significance.
        '''
        errors = 0
        tests = 200
        significance = 0.95
        for i in range(tests):
            df = generate_random_a_b(100, [0.5], [0.5])
            sase = SampleSet(df)
            p_val = sase.mutual_information(['a'], ['b'])['p_val']
            if p_val > significance:
                errors += 1

        std = np.sqrt((significance*(1-significance))/tests)
        self.assertAlmostEqual(1-significance, errors/tests, delta=3*std)

#    def test_mutual_information_significance_multi_category(self):
#        '''
#        Test that when generating A, B randomly according to P(A), P(B), the % of Type I errors matches the significance.
#        '''
#        errors = 0
#        tests = 200
#        significance = 0.95
#        for i in range(tests):
#            df = generate_random_a_b(100000, [0.1, 0.2, 0.4], [0.1, 0.3, 0.2, 0.1])
#            sase = SampleSet(df)
#            p_val = sase.mutual_information(['a'], ['b'], debug=True)['p_val']
#            if p_val > significance:
#                errors += 1
#
#        std = np.sqrt((significance*(1-significance))/tests)
#        self.assertAlmostEqual(1-significance, errors/tests, delta=3*std)
#
#    def test_mutual_information_significance_extreme(self):
#        '''
#        Test that when generating A, B randomly according to P(A), P(B), the % of Type I errors matches the significance.
#        '''
#        errors = 0
#        tests = 200
#        significance = 0.95
#        for i in range(tests):
#            df = generate_random_a_b(100000, [0.01], [0.99])
#            sase = SampleSet(df)
#            p_val = sase.mutual_information(['a'], ['b'], debug=True)['p_val']
#            if p_val > significance:
#                errors += 1
#
#        std = np.sqrt((significance*(1-significance))/tests)
#        self.assertAlmostEqual(1-significance, errors/tests, delta=3*std)
#
#    def test_mutual_information_significance_with_conditional(self):
#        '''
#        Test that when generating A, B randomly according to P(A), P(B), for a numbr of categories of a conditional group C, the % of Type I errors matches the significance.
#        '''
#        errors = 0
#        tests = 200
#        significance = 0.95
#        for i in range(tests):
#            final = pd.DataFrame({'a': [], 'b': [], 'c': []})
#            for j in [0, 5, 10]:
#                df = generate_random_a_b(10000, [0.05+j], [0.95-j])
#                df['c'] = j
#                final = final.append(df)
#            sase = SampleSet(final)
#            p_val = sase.mutual_information(['a'], ['b'], ['c'], debug=True)['p_val']
#            if p_val > significance:
#                errors += 1
#
#        std = np.sqrt((significance*(1-significance))/tests)
#        self.assertAlmostEqual(1-significance, errors/tests, delta=3*std)

if __name__ == '__main__':
    unittest.main()

