# This code was originally based off the MATLAB code in:
# Tucker McClure (2021). Numerical Inverse Laplace Transform (https://www.mathworks.com/matlabcentral/fileexchange/39035-numerical-inverse-laplace-transform), MATLAB Central File Exchange. Retrieved April 19, 2021.
# The original paper was:
# Abate, Joseph, and Ward Whitt. "A Unified Framework for Numerically Inverting Laplace Transforms." INFORMS Journal of Computing, vol. 18.4 (2006): 408-421. Print.

import numpy as np
from numpy import prod  # Do direct imports to match matlab functions exactly (i.e. prod(x) as opposed to np.prod(.))
from numpy import sum  # Warning, overrides the basic sum
from numpy import ones
from numpy import zeros
from numpy import log
from numpy import mod
from numpy import pi
from numpy import meshgrid
from numpy import real
import sympy


def euler_inversion(f_s, times, Marg=32):
    def bnml(n,z):
        one_to_z = np.arange(1, z+1)
        return prod( (n-(z-one_to_z))/one_to_z )

    #xi = np.array([0.5, ones(Marg), zeros(Marg-1), 2**-Marg])   # do ones(Marg), not ones(1,Marg) unlike matlab
    xi = np.concatenate( ([0.5], ones(Marg), zeros(Marg-1), [2**-Marg])  )
    for k in range(1,Marg):
        xi[2*Marg-k ] = xi[2*Marg-k + 1] + 2**-Marg * bnml(Marg,k)
    k = np.arange(0,2*Marg+1)
    beta = Marg*log(10)/3 + 1j*pi*k
    eta = (1-mod(k, 2)*2) * xi
    beta_mesh, t_mesh = meshgrid(beta, times)
    eta_mesh = meshgrid(eta, times)
    ilt = 10 ** (Marg/3) / times * sum (eta_mesh * real(f_s(beta_mesh / t_mesh)), axis=2)
    return ilt

