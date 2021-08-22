# Written by Rahul Yerrabelli
# This code was originally based off the MATLAB code in:
# Tucker McClure (2021). Numerical Inverse Laplace Transform (https://www.mathworks.com/matlabcentral/fileexchange/39035-numerical-inverse-laplace-transform), MATLAB Central File Exchange. Retrieved April 19, 2021.
# The original paper was:
# Abate, Joseph, and Ward Whitt. "A Unified Framework for Numerically Inverting Laplace Transforms." INFORMS Journal of Computing, vol. 18.4 (2006): 408-421. Print.
# Online version of the paper can be found at:
# http://www.columbia.edu/~ww2040/allpapers.html
# http://www.columbia.edu/~ww2040/AbateUnified2006.pdf

import numpy as np
from numpy import prod  # Do direct imports to match matlab functions exactly (i.e. prod(.) as opposed to np.prod(.))
from numpy import sum  # Warning, overrides the basic sum
from numpy import ones
from numpy import zeros
from numpy import log
from numpy import mod
from numpy import pi
from numpy import meshgrid
from numpy import real

import utils


def reload_imports():
    import importlib
    importlib.reload(inverting)
    importlib.reload(utils)


def euler_inversion(f_s, times, Marg=None):
    """
    For the comments, assume that N represents the length of the 1D array "times"
    """
    if Marg is None:
        Marg = 32

    def bnml(n, z):
        one_to_z = np.arange(1, z+1)
        return prod( (n-(z-one_to_z))/one_to_z )


    # f_s.shape = (N,)
    #xi = np.array([0.5, ones(Marg), zeros(Marg-1), 2**-Marg]) 
    xi = np.concatenate( ([0.5], ones(Marg), zeros(Marg-1), [2**-Marg])  )   # do ones(Marg), not ones(1,Marg) unlike matlab
    for k in range(1,Marg):
        xi[2*Marg-k ] = xi[2*Marg-k + 1] + 2**-Marg * bnml(Marg,k)
    # xi.shape = k.shape = beta.shape = eta.shape = (2*Marg+1,)
    k = np.arange(0,2*Marg+1)
    beta = Marg*log(10)/3 + 1j*pi*k
    eta = (1-mod(k, 2)*2) * xi
    # eta_mesh.shape = beta_mesh.shape = t_mesh.shape = (N, 2*Marg+1)
    beta_mesh, t_mesh = meshgrid(beta, times)
    eta_mesh, _ = meshgrid(eta, times)    # _ doesn't need to be saved as a variable as it should be the same as t_mesh
    try:
        # (beta_mesh / t_mesh).shape = (N, 2*Marg+1)
        f_s_val, is_inf = f_s(beta_mesh / t_mesh, return_error_inds=True)
        # ilt.shape = times.shape = (N,)
        ilt = 10 ** (Marg/3) / times * sum (eta_mesh * real(f_s_val), axis=1)
        
        (indices_is_inf, ) = np.nonzero(is_inf)
        print(f"Warning the function could not be inverted at some ({len(indices_is_inf)}/{len(is_inf)}) values of t as the I1(sqrt(f)) component "
              f"led to +/- infinity. The indices of these time points are {utils.abbreviate(indices_is_inf)}. The values are {times[indices_is_inf]}")
        
    except TypeError as exc:  # **function_name***() got an unexpected keyword argument 'return_error_inds'
        ilt = 10 ** (Marg/3) / times * sum (eta_mesh * real(f_s(beta_mesh / t_mesh)), axis=1)
        
    return ilt

