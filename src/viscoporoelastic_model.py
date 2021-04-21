import numpy as np
from numpy import exp
from numpy import sqrt
import scipy.special as sp

# Numpy besseli (i0) function doesn't support complex values and only has order 0
def I0(x): return sp.iv(0, x) #return np.i0(x); #besseli(0, x)
def I1(x): return sp.iv(1, x) #besseli(1, x)
def ln(x):
    return np.log(x)
    #import math
    #return math.log(x)


### Start combined matlab/python code



"""
## Fitted parameters (to be determined by experimental fitting to
# the unknown material)
c = 1;
tau1 = 1;
tau2 = 1;
#tau = [tau1, tau2];
#tau = [1 1];
tg=40.62; #in units of s   # for porosity_sp == 0.5
Vrtheta = 1; # Not actually v, but greek nu (represents Poisson's ratio)
Err = 1;
"""


class ViscoporoelasticModel:
    ## PARAMETERS
    ## Predefined constants
    eps0 = 0.1;  # 10 percent
    strain_rate = 0.1;  # 1 percent per s (normally 1#/s)
    ## Below are directly determined by the mesh deformation part of the
    ## experiment (see our paper with Daniel).  -Dr. Spector
    Vrz = 0.5;  # Not actually v, but greek nu (represents Poisson's ratio)
    Ezz = 10;  # Note- don't mix up Ezz with epszz


    def __init__(self):
        self.c = 1;
        self.tau1 = 1;
        self.tau2 = 1;
        # tau = [tau1, tau2];
        # tau = [1 1];
        self.tg = 40.62;  # in units of s   # for porosity_sp == 0.5
        self.Vrtheta = 1;  # Not actually v, but greek nu (represents Poisson's ratio)
        self.Err = 1;

    @staticmethod
    def predefined_constants():
        return ViscoporoelasticModel.eps0, ViscoporoelasticModel.strain_rate, ViscoporoelasticModel.Vrz, ViscoporoelasticModel.Ezz


    def laplace_value(self, s,
                         ## Fitted parameters (to be determined by experimental fitting to
                         # the unknown material)
                         c=1,
                         tau1 = 1,
                         tau2 = 1,    # tau = [tau1, tau2]; # tau = [1 1];
                         tg = 40.62,  # in units of s   # for porosity_sp == 0.5
                         Vrtheta = 1,  # Not actually v, but greek nu (represents Poisson's ratio)
                         Err = 1
                         ):
        print(self.c)
        c = self.c;
        tau1 = self.tau1;
        tau2 = self.tau2;
        # tau = [tau1, tau2];
        # tau = [1 1];
        tg = self.tg;  # in units of s   # for porosity_sp == 0.5
        Vrtheta = self.Vrtheta;  # Not actually v, but greek nu (represents Poisson's ratio)
        Vrtheta = self.Vrtheta;  # Not actually v, but greek nu (represents Poisson's ratio)
        Err = self.Err;

        eps0, strain_rate, Vrz, Ezz = ViscoporoelasticModel.predefined_constants()


        #print(s)
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
        return sigbar


        def test():
            import time as timer
            # inputting a value of time=0 doesn't error (just returns None/NaN), but takes longer (about 2x as much) on python; not really MATLAB though
            times=np.array([2,3])
            times = np.arange(0.05, 5.05, 0.05)
            from euler_inversion import euler_inversion
            t1=timer.time();
            ret=euler_inversion(self.laplace_function, times)
            print(ret)
            t2=timer.time()-t1
            print(t2)

