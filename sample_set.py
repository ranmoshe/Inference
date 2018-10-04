'''
A class for statistical calculations on a sample set.
Its input is a pandas.DataFrame whose columns are all of numpy.str type and are categorical.
'''

import numpy as np
import pandas as pd

class SampleSet():
    def __init__(self, matrix):
        self.matrix = matrix

    def prob_non_degenerated(self, columns):
        sub_matrix = self.matrix[columns]
        sub_matrix_count = sub_matrix.shape[0]
        joint_prob = sub_matrix.drop_duplicates()
        joint_prob = joint_prob.assign(joint_prob = lambda x: np.nan)

        for row in joint_prob.itertuples():
            column_values = {column:val for column,val in row._asdict().items() if column not in ['Index', 'joint_prob']}
            query = ' and '.join(['{column}=="{value}"'.format(column=column, value=value) for column, value in column_values.items()])
            row_count = sub_matrix.query(query).shape[0]
            row_joint_prob = row_count / sub_matrix_count
            joint_prob.at[row.Index, 'joint_prob'] = row_joint_prob
        
        return joint_prob

    def prob_degenerated(self):
        raw_data = {'joint_prob': [1]}
        return pd.DataFrame(raw_data, columns = ['joint_prob'])
        
    def probability(self, columns):
        '''
        Joint probability for a list of column names
        '''
        if not columns:
            return self.prob_degenerated()
        else:
            return self.prob_non_degenerated(columns)

    def entropy(self, columns):
        '''
        Entropy for a list of column names
        '''
        probablities = self.probability(columns)
        entropy = 0
        for row in probablities.itertuples():
            jp = row.joint_prob
            entropy += -jp*np.log2(jp)

        return entropy

    def A_given_B_value(self, groupA, groupB, B_val):
        vals = B_val.split(',')
        zipped = zip(groupB, vals)
        val_per_column = [x for x in zipped]
        queries = ['{col} =="{val}"'.format(col=x[0], val=x[1]) for x in val_per_column]
        query = ' & '.join(queries)
        return self.matrix.query(query)[groupA]

    @staticmethod
    def _get_query(columns, row):
        column_conditions = ['{col}=="{val}"'.format(col=key, val=val) for key,val in row._asdict().items() if key in columns]
        return ' and '.join(column_conditions)

    @staticmethod
    def _get_jp_for_query(jpTable, query):
        if not query:
            return float(jpTable.joint_prob)
        else:
            return float(jpTable.query(query).joint_prob)

    def _mi_for_row_values(self, row, groupA, groupB, groupConditional, probABC, probAC, probBC, probC):
        jp_ABC = float(row.joint_prob)
        query_AC = self._get_query(groupA+groupConditional, row)
        jp_AC = self._get_jp_for_query(probAC, query_AC)
        query_BC = self._get_query(groupB+groupConditional, row)
        jp_BC = self._get_jp_for_query(probAC, query_AC)
        query_C = self._get_query(groupConditional, row)
        jp_C = self._get_jp_for_query(probC, query_C)
        log = np.log2(jp_ABC*jp_C/(jp_AC*jp_BC))
        return jp_ABC*log

    def mutual_information(self, groupA, groupB, groupConditional=[]):
        '''
        I(A;B|C). https://en.wikipedia.org/wiki/Conditional_mutual_information
        '''
        probABC = self.probability(list(set(groupA+groupB+groupConditional)))
        probAC = self.probability(list(set(groupA+groupConditional)))
        probBC = self.probability(list(set(groupB+groupConditional)))
        probC = self.probability(groupConditional)
        mi = 0
        for row in probABC.itertuples():
            mi += self._mi_for_row_values(row, groupA, groupB, groupConditional, probABC, probAC, probBC, probC)

        return mi
