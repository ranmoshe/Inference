'''
A class for statistical calculations on a sample set.
Its input is a pandas.DataFrame whose columns are all of numpy.str type and are categorical.
'''

import numpy as np
import pandas as pd
from scipy.stats import chisquare

class SampleSet():
    def __init__(self, matrix):
        self.matrix = matrix
        self.count = self.matrix.shape[0]

    def prob_non_degenerated(self, columns):
        sub_matrix = self.matrix[columns]
        sub_matrix_count = sub_matrix.shape[0]
        joint_prob = sub_matrix.drop_duplicates()
        joint_prob = joint_prob.assign(joint_prob = lambda x: np.nan, row_count = lambda x: np.nan)

        for row in joint_prob.itertuples():
            column_values = {column:val for column,val in row._asdict().items() if column not in ['Index', 'joint_prob', 'row_count']}
            query = ' and '.join(['{column}=="{value}"'.format(column=column, value=value) for column, value in column_values.items()])
            row_count = sub_matrix.query(query).shape[0]
            row_joint_prob = row_count / sub_matrix_count
            joint_prob.at[row.Index, 'joint_prob'] = row_joint_prob
            joint_prob.at[row.Index, 'row_count'] = row_count
        
        return joint_prob

    def prob_degenerated(self):
        raw_data = {'joint_prob': [1], 'row_count': [self.matrix.shape[0]]}
        return pd.DataFrame(raw_data, columns = ['joint_prob', 'row_count'])
        
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

    def _likelihood(self, groupA, groupB, groupConditional, probABC, probAC, probBC):
        null_vals = [] # values for the null hypotheses
        observed_vals = []
        probC = self.probability(groupConditional)
        null_df = 0
        observed_df = 0
        # iterate over a|c values
        for ac_row in probAC.itertuples():
            conditional_null_df = 0
            conditional_observed_df = 0
            c_query = self._get_query(groupConditional, ac_row)
            a_query = self._get_query(groupA, ac_row)
            try:
                c_count = probC.query(c_query).iloc[0].row_count
            except ValueError:
                c_count = probC.iloc[0].row_count
            try:
                iterator = probBC.query(c_query).itertuples()
            except ValueError:
                iterator = probBC.itertuples()
            for bc_row in iterator:
                conditional_null_df += 1
                b_query = self._get_query(groupB, bc_row)
                null_vals.append(ac_row.joint_prob*bc_row.joint_prob*c_count)
                if c_query:
                    joint_query = ' and '.join([a_query, b_query, c_query])
                else:
                    joint_query = ' and '.join([a_query, b_query])
                try:
                    prob_abc = probABC.query(joint_query)
                    if not prob_abc.empty:
                        observed_vals.append(probABC.query(joint_query).iloc[0].row_count)
                        conditional_observed_df += 1
                    else:
                        observed_vals.append(0)
                except KeyError:
                    observed_vals.append(0)
            null_df += max(0, conditional_null_df -1)
            observed_df += max(0, conditional_observed_df -1)
        ddof = null_df - observed_df
        return chisquare(f_obs=observed_vals, f_exp=null_vals, ddof=ddof)

    @staticmethod
    def separate_categories(groupA, groupB, groupConditional):
        '''
        Test that columns in categories are separate, i.e. categories are mutually exclusive
        '''
        separate = True

        if list(set(groupA) & set(groupB)) != []:
            separate = False

        if list(set(groupA) & set(groupConditional)) != []:
            separate = False

        if list(set(groupB) & set(groupConditional)) != []:
            separate = False

        return separate


    def mutual_information(self, groupA, groupB, groupConditional=[]):
        '''
        I(A;B|C). https://en.wikipedia.org/wiki/Conditional_mutual_information
        '''
        if not self.separate_categories(groupA, groupB, groupConditional):
            raise Exception('No mutual values allowed between groups')
        probABC = self.probability(list(set(groupA+groupB+groupConditional)))
        probAC = self.probability(list(set(groupA+groupConditional)))
        probBC = self.probability(list(set(groupB+groupConditional)))
        probC = self.probability(groupConditional)
        mi = 0
        for row in probABC.itertuples():
            mi += self._mi_for_row_values(row, groupA, groupB, groupConditional, probABC, probAC, probBC, probC)

        likelihood = self._likelihood(groupA, groupB, groupConditional, probABC, probAC, probBC)

        return {'mi': mi, 'likelihood': likelihood}
