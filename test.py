# https://stackoverflow.com/questions/47278520/how-to-perform-a-numerical-laplace-and-inverse-laplace-transforms-in-matlab-for
# https://stackoverflow.com/questions/41042699/sympy-computing-the-inverse-laplace-transform


# import inverse_laplace_transform
import numpy as np
from sympy.integrals.transforms import inverse_laplace_transform
from sympy import exp, Symbol, lambdify
from sympy.abc import s, t
t = Symbol("t", positive=True)  # defining as positive simplifies inverse laplace
#s = Symbol("s", positive=True)
a = Symbol('a', positive=True)
# Using inverse_laplace_transform() method
gfg = inverse_laplace_transform(exp(-a * s) / s, s, t)

print(gfg)


def ln(x):
    import math
    return math.log(x)

#from scipy.special import i0 as I0
#from scipy.special import i1 as I1
#from math import sqrt

from sympy import sqrt  # Not math.sqrt as that would not accept expression input
from sympy.functions.special.bessel import besseli
def I0(x): return besseli(0, x)
def I1(x): return besseli(1, x)

### Start combined matlab/python code

## PARAMETERS

## Predefined constants
eps0 = 0.1; # 10 percent
strain_rate = 0.1; # 1 percent per s (normally 1#/s)
## Below are directly determined by the mesh deformation part of the
## experiment (see our paper with Daniel).  -Dr. Spector
Vrz = 0.5; # Not actually v, but greek nu (represents Poisson's ratio)
Ezz = 10;  # Note- don't mix up Ezz with epszz


## Fitted parameters (to be determined by experimental fitting to
# the unknown material)
c = 1;
tau1 = 1;
tau2 = 1;
#tau = [tau1 tau2];
#tau = [1 1];
tg=40.62; #in units of s   # for porosity_sp == 0.5
Vrtheta = 1; # Not actually v, but greek nu (represents Poisson's ratio)
Err = 1;






## BASE EQUATIONS
#  1
#eps0 = strain_rate * t0
t0 = eps0/strain_rate;
epszz = 1 - exp(-s*t0)/(s*s);  ##  Laplace transform of the axial strain



#  2
Srr     = 1/Err;
Srtheta = -Vrtheta/Err;
Srz     = -Vrz/Err;
Szz     = 1/Ezz;
#Sij     = [Srr, Srtheta, Srz;   Srtheta, Srr, Srz;   Srz, Srz, Szz];

#  3
alpha   =  2*Srz*Srz-Szz*Srtheta-Srr*Szz;
C13     =   Srz/(alpha);
C33     =  -(Srr+Srtheta)/(alpha);


#  4
g       =  -(2*Srz+Szz)*(Srr-Srtheta)/(alpha);

#  5
# Note- below could be simplified bc both divided and multiplied by 2
f1      =  -(2*Srz+Szz)/2 * 2*(Srr*Szz-Srz*Srz)/(alpha);

#  6
# Viscoelastic parameters: c, tau 1, tau 2
f2      = 1 + c*ln( (1+s*tau2)/(1+s*tau1) );

#  7
# Note- Ehat is a function of Sij although wasn't stated in Spector's notes
Ehat    =  -2*(Srr*Szz-Srz*Srz)/(alpha);


#  8
#f      =  r0^2*s / (Ehat*k*f2(c,tau1,tau2))
# Simplified using tg=r0^2/(Ehat*k)
# !!Confirm should be a function of c, tau also maybe Sij or tg
f       = tg * s/f2;



sigbar  =  \
    2*epszz*(\
        C13\
            *(\
                g \
                    * I1(sqrt(f))/sqrt(f) \
                    /(Ehat*I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f)) \
                -1/2 \
            ) \
        + C33/2 \
        + f1\
            *f2*\
            (I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f))\
            /(2 * ( Ehat*I0(sqrt(f)) - I1(sqrt(f))/sqrt(f) ) ) \
    );



#####
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

f_s = lambda new_s: sigbar.subs(s, new_s)
time=np.array(2)
time=np.array([2,3])
time = np.arange(0.2, 2, 0.2)
Marg=32

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
beta_mesh, t_mesh = meshgrid(beta, time)
eta_mesh = meshgrid(eta, time)
ilt = 10**(Marg/3)/time  * sum (eta_mesh * real( f_s(beta_mesh/t_mesh) ), axis=2 )
print(ilt)
