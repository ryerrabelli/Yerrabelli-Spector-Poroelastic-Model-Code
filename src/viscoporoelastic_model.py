import numpy as np
import scipy.optimize
from numpy import exp
from numpy import sqrt
import scipy.special as sp


# Numpy besseli (i0) function doesn't support complex values and only has order 0
def I0(x): return sp.iv(0, x) #return np.i0(x); #besseli(0, x)
def I1(x): return sp.iv(1, x) #besseli(1, x)
def J0(x): return sp.jv(0, x)
def J1(x): return sp.jv(1, x)
def ln(x): return np.log(x)  #import math #return math.log(x)


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
    def get_predefined_constant_names():
        return "eps0", "strain_rate", "Vrz", "Ezz"

    @staticmethod
    def get_predefined_constants():
        return ViscoporoelasticModel.eps0, ViscoporoelasticModel.strain_rate, ViscoporoelasticModel.Vrz, ViscoporoelasticModel.Ezz

    @staticmethod
    def get_fitted_parameter_names():
        return "c", "tau1", "tau2", "tg", "Vrtheta", "Err"

    # This is not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fitted_parameters(self):
        return self.c, self.tau1, self.tau2, self.tg, self.Vrtheta, self.Err;

    @staticmethod
    def get_var_categories():
        return ("Constant",)    * len(ViscoporoelasticModel.get_predefined_constant_names()) + \
               ("FittedParam",) * len(ViscoporoelasticModel.get_fitted_parameter_names())

    def set_fitted_parameters(self,
                          ## Fitted parameters (to be determined by experimental fitting to
                          # the unknown material)
                          c=None,
                          tau1=None,
                          tau2=None,  # tau = [tau1, tau2];
                          tg=None,  # in units of s   # for porosity_sp == 0.5
                          Vrtheta=None,  # Not actually v, but greek nu (represents Poisson's ratio)
                          Err=None,
                          ):
        if c is not None:
            self.c = c
        if tau1 is not None:
            self.tau1 = tau1
        if tau2 is not None:
            self.tau2 = tau2
        if tg is not None:
            self.tg = tg
        if Vrtheta is not None:
            self.Vrtheta = Vrtheta
        if Err is not None:
            self.Err = Err
        return self.get_fitted_parameters()

    def laplace_value(self,
                      s,
                      ## Fitted parameters (to be determined by experimental fitting to
                      # the unknown material)
                      c=None,
                      tau1=None,
                      tau2=None,  # tau = [tau1, tau2];
                      tg=None,  # in units of s   # for porosity_sp == 0.5
                      Vrtheta=None,  # Not actually v, but greek nu (represents Poisson's ratio)
                      Err=None,
                      ):

        """
        self.set_fitted_parameters(c=c, tau1=tau1, tau2=tau2, tg=tg, Vrtheta=Vrtheta, Err=Err)
        c = self.c;
        tau1 = self.tau1;
        tau2 = self.tau2;
        # tau = [tau1, tau2];
        # tau = [1 1];
        tg = self.tg;  # in units of s   # for porosity_sp == 0.5
        Vrtheta = self.Vrtheta;  # Not actually v, but greek nu (represents Poisson's ratio)
        Err = self.Err;
        """
        c, tau1, tau2, tg, Vrtheta, Err = self.set_fitted_parameters(c=c, tau1=tau1, tau2=tau2, tg=tg, Vrtheta=Vrtheta, Err=Err)

        eps0, strain_rate, Vrz, Ezz = ViscoporoelasticModel.get_predefined_constants()


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
            sigma=euler_inversion(self.laplace_function, times)
            print(sigma)
            t2=timer.time()-t1
            print(t2)


class TestModel:
    alpha = 0.5; tg = 7e-3; strain_rate = 1e-4; t0 = 1e3

    def laplace_value(self, s=None, alpha=None, tg=None, strain_rate=None, t0=None):  #, s, alpha=0.5, tg=7e-3, strain_rate=1e-4, t0=1e3
        if alpha is None:
            alpha = self.alpha
        if tg is None:
            tg = self.tg
        if strain_rate is None:
            strain_rate = self.strain_rate
        if t0 is None:
            t0 = self.t0

        eps  = tg*strain_rate*(1 - exp(-s*t0/tg))/(s*s);
        F = eps * (3*I0(sqrt(s))-8*alpha*I1(sqrt(s))/sqrt(s)) / (I0(sqrt(s))-2*alpha*I1(sqrt(s))/sqrt(s))
        return F


class TestModel2:
    vs = 0
    tg = 7e-3  # sec
    Es = 7e6  # Pa
    eps0 = 0.001
    a = 0.003  # meters

    @staticmethod
    def get_predefined_constants():
        return TestModel2.vs, TestModel2.tg, TestModel2.Es, TestModel2.eps0, TestModel2.a

    def laplace_value(self, s):
        vs, tg, Es, eps0, a = TestModel2.get_predefined_constants()
        eps = -eps0/s
        alpha = (1-2*vs)/(2*(1+vs))
        F = eps * (3*I0(sqrt(s))-8*alpha*I1(sqrt(s))/sqrt(s)) / (I0(sqrt(s))-2*alpha*I1(sqrt(s))/sqrt(s))
        return F

    def inverted_value(self, t, bessel_len=20):
        vs, tg, Es, eps0, a = TestModel2.get_predefined_constants()

        alpha2 = np.zeros(shape=bessel_len)
        for n in range(bessel_len):
            def eqn(x):
                return J1(x) - (1-vs)/(1-2*vs)*x*J0(x)
            # Use (n+1)*pi instead of n*pi bc python is zero-indexed unlike Matlab
            alpha2[n] = scipy.optimize.fsolve(func=eqn, x0=(n+1)*np.pi)

        summation = 0
        for n in range(bessel_len):
            temp = 1-2*vs
            An = (1-vs)*temp/(1+vs) * 1/(temp*temp*alpha2[n]-temp)
            summation += An * np.exp(-alpha2[n]*t/tg)

        F = np.pi * a
        F = np.pi * a*a
        F = np.pi * a*a * -Es
        F = np.pi * a*a * -Es * eps0
        F = np.pi * a*a * -Es * eps0 * (1 + summation)
        return F


