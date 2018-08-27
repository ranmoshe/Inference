'''
A class for statistical calculations on a sample set.
Its input is a pandas.DataFrame whose columns are all of numpy.str type and are categorical.
'''

import numpy as np

class SampleSet():
    def __init__(self, matrix):
        self.matrix = matrix

    def probability(self, columns):
        '''
        Joint probability for a list of column names
        '''
        sub_matrix = self.matrix[columns]
        combo_count = {}
        for sub_row in sub_matrix.values:
            combo = ','.join(sub_row)
            try:
                combo_count[combo] += 1
            except KeyError:
                combo_count[combo] = 1

        joint_probability = {}
        for combo, count in combo_count.items():
            joint_probability[combo] = count / len(sub_matrix.values)

        return joint_probability

    def entropy(self, columns):
        '''
        Entropy for a list of column names
        '''
        jp = self.probability(columns)
        entropy = 0
        for val in jp.values():
            entropy += -val*np.log2(val)

        return entropy

    def A_given_B_value(self, groupA, groupB, B_val):
        vals = B_val.split(',')
        zipped = zip(groupB, vals)
        val_per_column = [x for x in zipped]
        queries = ['{col} =="{val}"'.format(col=x[0], val=x[1]) for x in val_per_column]
        query = ' & '.join(queries)
        return self.matrix.query(query)[groupA]

    def mutual_information(self, groupA, groupB):
        '''
        Mutual information between columns in groupA and columns in groupB
        '''
        entropyA = self.entropy(groupA)
        probB = self.probability(groupB)
        mi = entropyA
        for B_category, B_probability in probB.items():
            subA = SampleSet(self.A_given_B_value(groupA, groupB, B_category))
            subA_entropy = subA.entropy(groupA)
            mi = mi - B_probability*subA_entropy

        return mi

