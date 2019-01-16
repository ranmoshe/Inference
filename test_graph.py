import numpy as np
import pandas as pd
import random
import unittest

from sample_set import SampleSet
from ic_graph import IC_Graph

class TestGraph(unittest.TestCase):

    def test_three_categories(self):
        '''
        We have a cat, an owner and a kitchen.
        The cat, left alone, messes up the kitchen.
        The owner tidies up, and also prevents the cat from messing up.
        The kitchen is either tidy or messy.
        Let's see if the graph captures that.
        '''
        pass

if __name__ == '__main__':
    unittest.main()

