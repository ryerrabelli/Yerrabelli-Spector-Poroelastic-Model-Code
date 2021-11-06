import unittest

import IPython.core.display

from viscoporoelastic_model import TestModel2, ViscoporoelasticModel1, CohenModel
import inverting
import numpy as np
import mpmath

class MyTestCase(unittest.TestCase):
    def setUp(self): # This is a special unittest method
        self.tm2 = TestModel2()
        self.vpe1 = ViscoporoelasticModel1()

    def test_cohen(self, bessel_len=50):
        vpe = CohenModel()
        time_const = vpe.tg
        func = [vpe.laplace_value for vpe, label in [(vpe, fr"$Cohen$")]]
        input_times = np.linspace(0.001, 2, num=1001, endpoint=True) * time_const

        input_times = np.array([0.05, 0.1, 0.25, 0.5, 3]) * time_const

        results = {
            "t": input_times,
            "t/tg": input_times / time_const,
            "Analytic": vpe.inverted_value(t=input_times)
        }
        inversion_methods = {
            #"euler": lambda f, t: inverting.euler_inversion(f, t, Marg=None),
            "euler64": lambda f, t: inverting.euler_inversion(f, t, Marg=64),
            "euler56": lambda f, t: inverting.euler_inversion(f, t, Marg=56),
            "euler48": lambda f, t: inverting.euler_inversion(f, t, Marg=48),
            #"euler40": lambda f, t: inverting.euler_inversion(f, t, Marg=40),
            "euler32": lambda f, t: inverting.euler_inversion(f, t, Marg=32),
            #"euler16": lambda f, t: inverting.euler_inversion(f, t, Marg=16),
            #"mpmath": lambda f, t: mpmath.invertlaplace(f, t),
        }
        for name, invert in inversion_methods.items():
            while name in results.keys():
                name += "_"  # prevent duplicates
            print(f"Going to start {name} method")
            results[name] = invert(lambda s: func[0](s), input_times / time_const)

        import pandas as pd
        df3 = pd.DataFrame(results)
        df3.set_index("t")
        IPython.core.display.display(df3)





if __name__ == '__main__':
    unittest.main()
