import unittest

from src.viscoporoelastic_model import TestModel2
import numpy as np


class MyTestCase(unittest.TestCase):

    def setUp(self): # This is a special unittest method
        self.tm2 = TestModel2()

    def test_value_inversion(self):
        print( self.tm2.inverted_value(1.0/10000) )
        #print( self.tm2.inverted_value(np.arange(0.05, 5.05, 1)) )
        #print( self.tm2.inverted_value(np.arange(1,100,10)) )
        print( self.tm2.inverted_value(np.arange(1,10,1)/10000.0) )



if __name__ == '__main__':
    unittest.main()
