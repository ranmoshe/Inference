'''
Suite of tests for the module
'''
import unittest
import test_mutual_information
import test_graph


def main():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromModule(test_mutual_information))
    suite.addTests(loader.loadTestsFromModule(test_graph))

    runner = unittest.TextTestRunner(verbosity=3)
    runner.run(suite)

if __name__ == '__main__':
    main()

