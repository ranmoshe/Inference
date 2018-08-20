'''
Description of the program
'''

class SampleSet():
    def __init__(self, matrix):
        self.matrix = matrix

    def get_probability(self, columns):
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


        
