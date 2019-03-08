import numpy as np
import pandas as pd
import random
import unittest

from ic_graph import IC_Graph
from sample_set import SampleSet
from utils import generate_random_a_b, generate_random_series

class TestGraph(unittest.TestCase):

    @staticmethod
    def kitchen_state(row):
        if row['cat'] == 'near':
            choice = random.choices(
                    population=['messy', 'tidy'],
                    weights = [0.8, 0.2],
                    k = 1)[0]
        else:
            choice = random.choices(
                    population=['messy', 'tidy'],
                    weights = [0.1, 0.9],
                    k = 1)[0]

        if choice == 'messy':
            if row['owner'] == 'near':
                choice = random.choices(
                    population=['messy', 'tidy'],
                    weights = [0.25, 0.75],
                    k = 1)[0]
            else:
                choice = random.choices(
                    population=['messy', 'tidy'],
                    weights = [0.8, 0.2],
                    k = 1)[0]

        return choice


    def test_v_structure(self):
        '''
        We have a cat, an owner and a kitchen.
        The cat, left alone, messes up the kitchen.
        The owner tidies up, and also prevents the cat from messing up.
        The kitchen is either tidy or messy.
        Let's see if the graph captures that.
        '''
        df = generate_random_a_b(1000, [0.3], [0.2], a_name='cat', b_name='owner')
        df['cat'] = df['cat'].map({'cat_0': 'near', 'cat_1': 'far'})
        df['owner'] = df['owner'].map({'owner_0': 'near', 'owner_1': 'far'})
        df['kitchen'] = df.apply(lambda row: self.kitchen_state(row), axis=1)
        ic_graph = IC_Graph(SampleSet(df))
        ic_graph.build_graph()
        directed = [t for t in ic_graph.graph.edges.data('out')]

        self.assertEqual(
                [('cat', 'kitchen', 'kitchen'), ('owner', 'kitchen', 'kitchen')],
                directed
                )
        self.assertEqual(len(ic_graph.graph.edges), 2)
        
    @staticmethod
    def hunted_state(hunter_state, rest_if_rest, rest_if_run):
        if hunter_state == 'resting':
            weights = [rest_if_rest, 1-rest_if_rest]
        else:  # hunter_state == 'running'
            weights = [rest_if_run, 1-rest_if_run]

        choice = random.choices(
                population=['resting', 'running'],
                weights = weights,
                k = 1)[0]

        return choice

    def test_3_node_chain(self):
        '''
        We have a dog, a cat and a mouse. 
        Sometimes the dog feels like chasing the cat, and then the cat would mostly run. 
        If the cat is left alone, it can chase the mouse, and then the mouse would mostly run.
        The algorithm can't determine causal relationship by this data, so we expect an undirected chain.
        '''
        dog_series = generate_random_series(1000, 'dog', [0.2])
        df = pd.DataFrame({'dog': dog_series})
        df['dog'] = df['dog'].map({'dog_0': 'resting', 'dog_1': 'running'})
        df['cat'] = df.apply(lambda row: self.hunted_state(row['dog'], 0.8, 0.2), axis=1)
        df['mouse'] = df.apply(lambda row: self.hunted_state(row['cat'], 0.8, 0.2), axis=1)
        ic_graph = IC_Graph(SampleSet(df))
        ic_graph.build_graph()
        directed = [t for t in ic_graph.graph.edges.data('out') if t[2] is not None]

        self.assertEqual(
                [],
                directed
                )
        self.assertEqual(
                [('dog', 'cat'), ('cat', 'mouse')], 
                [t for t in ic_graph.graph.edges])
        

if __name__ == '__main__':
    unittest.main()

