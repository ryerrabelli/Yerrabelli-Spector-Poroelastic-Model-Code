# https://stackoverflow.com/questions/47278520/how-to-perform-a-numerical-laplace-and-inverse-laplace-transforms-in-matlab-for
# https://stackoverflow.com/questions/41042699/sympy-computing-the-inverse-laplace-transform

# import inverse_laplace_transform
from sympy.integrals.transforms import inverse_laplace_transform
from sympy import exp, Symbol
from sympy.abc import s, t

a = Symbol('a', positive=True)
# Using inverse_laplace_transform() method
gfg = inverse_laplace_transform(exp(-a * s) / s, s, t)

print(gfg)