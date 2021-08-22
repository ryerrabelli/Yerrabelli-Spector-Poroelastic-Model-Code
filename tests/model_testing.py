import unittest

from src.viscoporoelastic_model import TestModel2, ViscoporoelasticModel1
import numpy as np


class MyTestCase(unittest.TestCase):

    def setUp(self): # This is a special unittest method
        self.tm2 = TestModel2()
        self.vpe1 = ViscoporoelasticModel1()

    def test_value_inversion(self):
        print( self.tm2.inverted_value(1.0/10000) )
        #print( self.tm2.inverted_value(np.arange(0.05, 5.05, 1)) )
        #print( self.tm2.inverted_value(np.arange(1,100,10)) )
        print( self.tm2.inverted_value(np.arange(1,10,1)/10000.0) )


        s=0.01
        output = s*self.vpe1.laplace_value(s)
        print(output)





if __name__ == '__main__':
    unittest.main()
