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

    def _get_null_df_for_c(self, groupA, groupB):
        return len(groupA) - 1 + len(groupB) - 1

    def _get_null_df(self, groupConditional, probABC, probAC, probBC):
        df = 0
        if groupConditional:
            df = 0
        else:
            df += self._get_null_df_for_c(probAC, probBC)
            
        return df

    def _get_observed_df(self, groupConditional, probABC):
        if groupConditional:
            raise Exception('Not implemented yet')
        else:
            c_count = 1
        return len(probABC) - c_count

    @classmethod
    def _observed_vs_null(cls, groupA, groupB, matrix):
        null = []
        observed = []
        df_observed = 0
        total_rows = len(matrix)
        a_vals = matrix[groupA].drop_duplicates()
        b_vals = matrix[groupB].drop_duplicates()
        df_null = len(a_vals) + len(b_vals) - 2
        for a_val in a_vals.itertuples():
            a_query = cls._get_query(groupA, a_val)
            a_rows = len(matrix.query(a_query))
            for b_val in b_vals.itertuples():
                b_query = cls._get_query(groupB, b_val)
                joint_query = f"{a_query} and {b_query}"
                joint_matrix = matrix.query(joint_query)
                b_rows = len(matrix.query(b_query))
                null.append((b_rows/total_rows)*(a_rows/total_rows))
                if len(joint_matrix) > 0:
                    df_observed += 1
                    observed.append(len(joint_matrix))
                else:
                    observed.append(0)
        df_observed = df_observed - 1

        return (null, observed, df_null, df_observed)


    def _p_val(self, groupA, groupB, groupConditional, probABC, probAC, probBC, debug):
        if groupConditional:
            c_vals = self.matrix[groupConditional].drop_duplicates()
            for c_val in c_vals.itertuples():
                query = self._get_query(groupConditional, c_val)
                matrix_for_c_val = self.matrix.query(query)
                a_b_columns = list(set(groupA+groupB))
                matrix_for_c_val = matrix_for_c_val[a_b_columns]
                self._observed_vs_null(groupA, groupB, matrix_for_c_val)

            return 0
        else:
            null_vals, observed_vals, null_df, observed_df = self._observed_vs_null(groupA, groupB, self.matrix)
        final_dof = abs(null_df - observed_df)
        ddof = len(observed_vals) - 1 - final_dof
        res = chisquare(f_obs=observed_vals, f_exp=null_vals, ddof=ddof)[1]
        return res

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


    def mutual_information(self, groupA, groupB, groupConditional=[], debug=False):
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

        p_val = self._p_val(groupA, groupB, groupConditional, probABC, probAC, probBC, debug)

        return {'mi': mi, 'p_val': p_val}
