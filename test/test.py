# To run tests:
# python -m unittest discover test

import sys
import unittest
sys.path.append('.') 
sys.path.append('../') 

class TestSample(unittest.TestCase):

    def test_sample(self):
        self.assertTrue(True)
        PING = "PONG"
        self.assertEqual(PING, "PONG")



import numpy as np
from sslplay.utils.ssplit import ssplit

class TestUtils(unittest.TestCase):

    def test_ssplit(self):
        
        X = np.random.rand(1000, 10)
        y = np.random.choice([0, 1], size=X.shape[0], replace=True)
        
        X1, y1, X2, y2 = ssplit(X=X, y=y, percentage_1=50, percentage_2=50, seed=1102)
        self.assertTrue(len(y1) + len(y2) == len(y))
        self.assertTrue(X1.shape[0] == len(y1))
        self.assertTrue(X2.shape[0] == len(y2))
        self.assertTrue(len(np.unique(y1)) == len(np.unique(y2)))
        
        X3, y3, X4, y4 = ssplit(X=X, y=y, percentage_1=50, percentage_2=50, seed=1102)
        self.assertTrue(np.sum(y1 == y3) == len(y1))
        self.assertTrue(np.sum(y2 == y4) == len(y2))
        self.assertTrue(np.abs(len(y1) / len(y) - 0.5) < 0.1)
        
        X1, y1, X2, y2 = ssplit(X=X, y=y, percentage_1=99, percentage_2=1, seed=1102)
        self.assertTrue(len(np.unique(y1)) == len(np.unique(y2)))
        self.assertTrue(np.abs(len(y2) / len(y) - 0.01) < 0.01)




if __name__ == '__main__':
    unittest.main()