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

    def season_to_flu_epidemic(self, season):
        if season == 'winter':
            weights = [0.05, 0.95]
        elif season == 'spring':
            weights = [0.005, 0.995]
        elif season == 'summer':
            weights = [0.0005, 0.9995]
        elif season == 'fall':
            weights = [0.03, 0.97]

        choice = random.choices(
                population=['yes', 'no'],
                weights = weights,
                k = 1)[0]

        return choice

    def season_to_temperature(self, season):
        if season == 'winter':
            weights = [0.6, 0.3, 0.1]
        elif season == 'spring':
            weights = [0.15, 0.6, 0.25]
        elif season == 'summer':
            weights = [0.1, 0.3, 0.6]
        elif season == 'fall':
            weights = [0.25, 0.6, 0.15]

        choice = random.choices(
                population=['cold', 'mild', 'hot'],
                weights = weights,
                k = 1)[0]

        return choice

    def test_common_cause(self):
        '''
        Season affects both the temperature and the chance for a flu epidemic.
        The algorithm can't determine causal relationship by this data, so we expect an undirected chain.
        '''

        season_series = generate_random_series(10000, 'season', [0.35, 0.15, 0.35])
        df = pd.DataFrame({'season': season_series})
        df['season'] = df['season'].map({'season_0': 'winter', 'season_1': 'spring','season_2': 'summer', 'season_3': 'fall'})
        df['temperature'] = df.apply(lambda row: self.season_to_temperature(row['season']), axis=1)
        df['flu_epidemic'] = df.apply(lambda row: self.season_to_flu_epidemic(row['season']), axis=1)
        ic_graph = IC_Graph(SampleSet(df))
        ic_graph.build_graph()
        directed = [t for t in ic_graph.graph.edges.data('out') if t[2] is not None]
        import ipdb; ipdb.set_trace()

        self.assertEqual(
                [],
                directed
                )
        self.assertEqual(
                [('season', 'temperature'), ('season', 'flu_epidemic')], 
                [t for t in ic_graph.graph.edges])
     
    def test_3_node_chain(self):
        '''
        We have a dog, a cat and a mouse. 
        Sometimes the dog feels like chasing the cat, and then the cat would mostly run. 
        If the cat is left alone, it can chase the mouse, and then the mouse would mostly run.
        The algorithm can't determine causal relationship by this data, so we expect an undirected chain.
        '''
        dog_series = generate_random_series(10000, 'dog', [0.2])
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

