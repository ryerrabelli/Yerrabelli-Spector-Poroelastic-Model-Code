import unittest

from src.viscoporoelastic_model import TestModel2, ViscoporoelasticModel1, CohenModel
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

    def test_cohen(self, bessel_len=50):
        import scipy
        model = CohenModel()
        alpha2_vals = np.zeros(shape=bessel_len)
        for n in range(len(alpha2_vals)):
            alpha2_vals[n] = (scipy.optimize.fsolve(func=model.characteristic_eqn, x0=(n + 1) * np.pi))
            print(f"alpha_{n+1} = {alpha2_vals[n]:0.3f}")

        print([f"{(n+1)}*pi={(n+1)*np.pi:0.2f} => {alpha2_N:0.2f}" for n,alpha2_N in enumerate(alpha2_vals)])

        model.inverted_value(t=9.9)







if __name__ == '__main__':
    unittest.main()
