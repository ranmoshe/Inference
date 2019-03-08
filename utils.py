'''
Util methods for the module
'''
import pandas as pd
import random

def generate_random_series(rows, column_name, prob_array):
    category_indices = [str(i) for i in range(len(prob_array)+1)]
    population = [f"{column_name}_{i}" for i in category_indices]
    complement_weight = 1 - sum(prob_array)
    prob_array.append(complement_weight)
    return random.choices(
            population = population,
            weights = prob_array,
            k = rows)


def generate_random_a_b(rows, p_a, p_b, a_name='a', b_name='b'):
    a = generate_random_series(rows, a_name, p_a)
    b = generate_random_series(rows, b_name, p_b)
    df = pd.DataFrame({a_name: a, b_name: b})
    return df

def main():
    pass

if __name__ == '__main__':
    main()

