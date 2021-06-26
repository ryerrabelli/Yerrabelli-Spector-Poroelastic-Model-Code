import numpy as np
import scipy.optimize
from numpy import exp
from numpy import sqrt
import scipy as sp
import abc


# Numpy besseli (i0) function doesn't support complex values and only has order 0
def I0(x): return sp.special.iv(0, x) #return np.i0(x); #besseli(0, x)
def I1(x): return sp.special.iv(1, x) #besseli(1, x)
def J0(x): return sp.special.jv(0, x)
def J1(x): return sp.special.jv(1, x)
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


class LaplaceModel(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_predefined_constants(cls): raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def get_predefined_constant_names(): raise NotImplementedError()

    def get_parameters(self):
        return ()  # zero-length tuple, aka tuple()

    @staticmethod
    def get_parameter_names():
        return ()  # zero-length tuple, aka tuple()

    def get_calculable_constants(self):
        return ()  # zero-length tuple, aka tuple()

    @staticmethod
    def get_calculable_constant_names():
        return ()  # zero-length tuple, aka tuple()

    @abc.abstractmethod
    def laplace_value(self, s): return NotImplemented

    @classmethod
    def inverted_value_units(cls): return NotImplemented #return "N/A"

    def get_all_names_and_vars(self):
        tm = self

        #dict(zip(type(tm).get_predefined_constant_names(), tm.get_predefined_constants()))
        #dict(zip(type(tm).get_parameter_names(), tm.get_parameters()))
        #dict(zip(type(tm).get_calculable_constant_names(), tm.get_calculable_constants()))
        """return dict(zip(
            type(tm).get_predefined_constant_names() + type(tm).get_parameter_names() + type(
                tm).get_calculable_constant_names(),
            tm.get_predefined_constants() + tm.get_parameters() + tm.get_calculable_constants()
        ))"""

        return dict(zip(
            sum(
                [
                    tm.get_predefined_constant_names(),
                    tm.get_parameter_names(),
                    tm.get_calculable_constant_names(),
                ],
                tuple()  # "start" has to be an empty tuple (default is int 0, which throws an error when with tuples)
            ),
            sum(
                [
                    tm.get_predefined_constants(),
                    tm.get_parameters(),
                    tm.get_calculable_constants(),
                ],
                tuple()  # "start" has to be an empty tuple (default is int 0, which throws an error when with tuples)
            ),
        ))

    @classmethod
    def get_model_name(cls):
        # self.__class__.__name__
        # type(self).__name__
        return cls.__name__


class AnalyticallyInvertableModel(LaplaceModel, abc.ABC):
    def inverted_value(self, t): return NotImplemented


class ViscoporoelasticModel0(LaplaceModel):
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

    @classmethod
    def get_predefined_constants(cls):
        return cls.eps0, cls.strain_rate, cls.Vrz, cls.Ezz
        #return type(self).eps0, type(self).strain_rate, type(self).Vrz, type(self).Ezz

    @staticmethod
    def get_predefined_constant_names():
        return "eps0", "strain_rate", "Vrz", "Ezz"

    # This is not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fitted_parameters(self):
        return self.c, self.tau1, self.tau2, self.tg, self.Vrtheta, self.Err;

    @staticmethod
    def get_fitted_parameter_names():
        return "c", "tau1", "tau2", "tg", "Vrtheta", "Err"

    def get_parameters(self): return self.get_fitted_parameters()

    @classmethod
    def get_parameter_names(cls): return cls.get_fitted_parameter_names()

    @classmethod
    def get_var_categories(cls):
        return ("Constant",)    * len(cls.get_predefined_constant_names()) + \
               ("FittedParam",) * len(cls.get_fitted_parameter_names())

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

        eps0, strain_rate, Vrz, Ezz = self.get_predefined_constants()


        #print(s)
        ## BASE EQUATIONS
        #  1
        #eps0 = strain_rate * t0
        t0 = eps0/strain_rate;
        # TODO: Confirm the TestModel4 epszz expression from Dr. Spector as this seems to be different from the one
        #  for the viscoporoelastic model
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


class TestModel1(LaplaceModel):
    alpha = 0.5; tg = 7e-3; strain_rate = 1e-4; t0 = 1e3

    @classmethod
    def get_predefined_constants(cls):
        return cls.alpha, cls.tg, cls.strain_rate, cls.t0

    @staticmethod
    def get_predefined_constant_names():
        return "alpha", "tg", "strain_rate", "t0"

    def laplace_value(self, s, alpha=None, tg=None, strain_rate=None, t0=None):  #, s, alpha=0.5, tg=7e-3, strain_rate=1e-4, t0=1e3
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


class TestModel2(AnalyticallyInvertableModel):
    """
    vs = 0
    tg = 7e3  # sec
    Es = 7e6  # Pa
    eps0 = 0.001
    a = 0.003  # meters
    alpha2_vals=None
    A_vals=None
    saved_bessel_len = 0
    """

    def __init__(self):
        self.vs = 0;
        self.tg = 7e3  # sec
        self.Es = 7e6  # Pa
        self.eps0 = 0.001
        self.a = 0.003  # meters

        self.alpha2_vals = None
        self.A_vals = None
        self.saved_bessel_len = 0

    #@staticmethod
    #def get_predefined_constants():
    #    return TestModel2.vs, TestModel2.tg, TestModel2.Es, TestModel2.eps0, TestModel2.a
    @classmethod
    def get_predefined_constants(cls):
        return ()   # zero-length tuple

    @staticmethod
    def get_predefined_constant_names():
        return ()   # zero-length tuple

    def get_parameters(self):
        return self.vs, self.tg, self.Es, self.eps0, self.a

    @classmethod
    def get_parameter_names(cls):
        return "vs", "tg", "Es", "eps0", "a"

    def get_calculable_constants(self):
        vs, tg, Es, eps0, a = self.get_parameters()  #type(self).get_predefined_constants()
        alpha = (1-2*vs)/(2*(1+vs))
        return alpha,  # 1-length tuple

    @staticmethod
    def get_calculable_constant_names():
        return "alpha",  # 1-length tuple

    def characteristic_eqn(self, x):
        vs, tg, Es, eps0, a = self.get_parameters()
        return J1(x) - (1 - vs) / (1 - 2 * vs) * x * J0(x)

    def setup_constants(self, bessel_len=20):
        vs, tg, Es, eps0, a = self.get_parameters()
        alpha2_vals = np.zeros(shape=bessel_len)
        for n in range(bessel_len):
            # Use (n+1)*pi instead of n*pi bc python is zero-indexed unlike Matlab
            alpha2_vals[n] = scipy.optimize.fsolve(func=self.characteristic_eqn, x0=(n + 1) * np.pi)

        A_vals = np.zeros(shape=bessel_len)
        for n in range(bessel_len):
            temp = 1 - 2 * vs
            A_vals[n] = (1 - vs) * temp / (1 + vs) * 1 / (temp * temp * alpha2_vals[n] - temp)

        self.alpha2_vals=alpha2_vals
        self.A_vals = A_vals
        self.saved_bessel_len = bessel_len

    def laplace_value(self, s):
        vs, tg, Es, eps0, a = self.get_parameters()  #type(self).get_predefined_constants()
        alpha, = self.get_calculable_constants()

        eps = -eps0/s
        F = eps * (3*I0(sqrt(s))-8*alpha*I1(sqrt(s))/sqrt(s)) / (I0(sqrt(s))-2*alpha*I1(sqrt(s))/sqrt(s))
        return F

    def inverted_value(self, t, bessel_len=20):
        vs, tg, Es, eps0, a = self.get_parameters()

        if bessel_len > self.saved_bessel_len:
            self.setup_constants(bessel_len=bessel_len)
        A_vals = self.A_vals
        alpha2_vals = self.alpha2_vals

        summation = 0
        for n in range(bessel_len):
            summation += A_vals[n] * np.exp(-alpha2_vals[n]*t/tg)

        """
        F = np.pi * a
        F = np.pi * a*a
        F = np.pi * a*a * -Es
        F = np.pi * a*a * -Es * eps0
        F = np.pi * a*a * -Es * eps0 * (1 + summation)
        """
        F = np.pi * a*a * -Es * eps0 * (1 + summation)
        return F

    @classmethod
    def inverted_value_units(cls):
        return "Newtons"  # Newtons


class TestModel3(TestModel2):
    #@staticmethod
    #def characteristic_eqn(*args, **kwargs): return TestModel2.characteristic_eqn(*args, **kwargs)

    def laplace_value(self, s):
        """
        Overrides super function
        :param s:
        :return:
        """
        vs, tg, Es, eps0, a = self.get_parameters()  #TestModel3.get_predefined_constants()
        alpha, = self.get_calculable_constants()

        eps = -eps0/s
        U_a = -eps/2 * (I0(sqrt(s))-4*alpha*I1(sqrt(s))/sqrt(s)) / (I0(sqrt(s))-2*alpha*I1(sqrt(s))/sqrt(s))
        return U_a

    def inverted_value(self, t, bessel_len=20):
        """
        Overrides super function
        :return:
        """
        vs, tg, Es, eps0, a = self.get_parameters() #type(self).get_predefined_constants()

        if bessel_len > self.saved_bessel_len:
            self.setup_constants(bessel_len=bessel_len)
        A_vals = self.A_vals
        alpha2_vals = self.alpha2_vals

        summation_a = 0
        for n in range(bessel_len):
            summation_a += np.exp(-alpha2_vals[n]*t/tg)/(alpha2_vals[n]-1)

        return summation_a

    @classmethod
    def inverted_value_units(cls):
        return "unitless"  # displacement/a is m/m = unitless


class TestModel4(LaplaceModel):   # Dr. Spector sent this to me May 29, 2021
    """
    v = 0
    strain_rate = 0.0003  #1e-3  # s^-1
    t0_tg = 0.1
    tg = 1000  #7e3  # sec
    """

    def __init__(self):
        self.v = 0;
        self.strain_rate = 0.0003  # 1e-3  # s^-1
        self.t0_tg = 0.1
        self.tg = 1000  # 7e3  # sec

    @classmethod
    def get_predefined_constants(cls):
        # cls.v, cls.strain_rate, cls.t0_tg, cls.tg
        return ()   # zero-length tuple

    @staticmethod
    def get_predefined_constant_names():
        return ()   # zero-length tuple

    def get_parameters(self):
        return self.v, self.strain_rate, self.t0_tg, self.tg

    @classmethod
    def get_parameter_names(cls):
        return "v", "strain_rate", "t0/tg", "tg"

    def get_calculable_constants(self):
        v, strain_rate, t0_tg, tg = self.get_parameters()
        t0 = t0_tg * tg
        eps0 = strain_rate * t0
        C0 = (1-2*v)/(1-v)
        return t0, eps0, C0

    @staticmethod
    def get_calculable_constant_names():
        return "t0", "eps0", "C0"

    def laplace_value(self, s):
        """
        Overrides super function
        :param s:
        :return:
        """
        v, strain_rate, t0_tg, tg = self.get_parameters()
        t0, eps0, C0 = self.get_calculable_constants()

        # TODO: Confirm the TestModel4 epszz expression from Dr. Spector as this seems to be different from the one
        #  for the viscoporoelastic model

        epszz = eps0*(1 - exp(-s*t0/tg))/(s*s);  ##  Laplace transform of the axial strain
        f_prime = epszz * (3*I0(sqrt(s))-4*C0*I1(sqrt(s))/sqrt(s)) / (I0(sqrt(s))-C0*I1(sqrt(s))/sqrt(s))
        return f_prime


class ArmstrongIsotropicModel(LaplaceModel):   # Dr. Spector sent this to me May 29, 2021, then revised it on Jun 11, 2021
    """
    v = 0
    strain_rate = 0.0003  #1e-3  # s^-1
    t0_tg = 0.1
    tg = 1000  #7e3  # sec
    """

    def __init__(self):
        self.v = 0;
        self.strain_rate = 1e-4  # s^-1
        self.t0_tg = 100/7e3
        self.tg = 7e3  # sec

    @classmethod
    def get_predefined_constants(cls):
        # cls.v, cls.strain_rate, cls.t0_tg, cls.tg
        return ()   # zero-length tuple

    @staticmethod
    def get_predefined_constant_names():
        return ()   # zero-length tuple

    def get_parameters(self):
        return self.v, self.strain_rate, self.t0_tg, self.tg

    @classmethod
    def get_parameter_names(cls):
        return "v", "strain_rate", "t0/tg", "tg"

    def get_calculable_constants(self):
        v, strain_rate, t0_tg, tg = self.get_parameters()
        t0 = t0_tg * tg
        eps0 = strain_rate * t0
        C0 = (1-2*v)/(1-v)
        return t0, eps0, C0

    @staticmethod
    def get_calculable_constant_names():
        return "t0", "eps0", "C0"

    def laplace_value(self, s):
        """
        Overrides super function
        :param s:
        :return:
        """
        v, strain_rate, t0_tg, tg = self.get_parameters()
        t0, eps0, C0 = self.get_calculable_constants()

        # TODO: Confirm the TestModel4 epszz expression from Dr. Spector as this seems to be different from the one
        #  for the viscoporoelastic model

        epszz = strain_rate*tg*(1 - exp(-s*t0/tg))/(s*s);  ##  Laplace transform of the axial strain
        f_prime = epszz * (3*I0(sqrt(s))-4*C0*I1(sqrt(s))/sqrt(s)) / (I0(sqrt(s))-C0*I1(sqrt(s))/sqrt(s))
        return f_prime


class ViscoporoelasticModel1(LaplaceModel):
    ## PARAMETERS
    ## Predefined constants
    t0_tg = 0.1;
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

    @classmethod
    def get_predefined_constants(cls):
        return cls.t0_tg, cls.strain_rate, cls.Vrz, cls.Ezz
        #return type(self).eps0, type(self).strain_rate, type(self).Vrz, type(self).Ezz

    @staticmethod
    def get_predefined_constant_names():
        return "t0/tg", "strain_rate", "Vrz", "Ezz"

    # This is not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fitted_parameters(self):
        return self.c, self.tau1, self.tau2, self.tg, self.Vrtheta, self.Err;

    @staticmethod
    def get_fitted_parameter_names():
        return "c", "tau1", "tau2", "tg", "Vrtheta", "Err"

    def get_parameters(self): return self.get_fitted_parameters()

    @classmethod
    def get_parameter_names(cls): return cls.get_fitted_parameter_names()

    @classmethod
    def get_var_categories(cls):
        return ("Constant",)    * len(cls.get_predefined_constant_names()) + \
               ("FittedParam",) * len(cls.get_fitted_parameter_names())

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

        t0_tg, strain_rate, Vrz, Ezz = self.get_predefined_constants()


        #print(s)
        ## BASE EQUATIONS
        #  2 (<-1)
        # Below lines modified from March to June 2021 versions
        t0 = t0_tg * tg
        #eps0 = strain_rate * t0
        epszz = strain_rate * tg * (1 - exp(-s*t0/tg)/(s*s));  ##  Laplace transform of the axial strain



        #  3 (<-2)
        Srr     = 1/Err;
        Srtheta = -Vrtheta/Err;
        Srz     = -Vrz/Err;
        Szz     = 1/Ezz;
        #Sij     = [Srr, Srtheta, Srz;   Srtheta, Srr, Srz;   Srz, Srz, Szz];

        #  4 (<-3)
        alpha   =  2*Srz*Srz-Szz*Srtheta-Srr*Szz;
        C13     =   Srz/(alpha);
        C33     =  -(Srr+Srtheta)/(alpha);
        # Note- Ehat is a function of Sij although wasn't stated in Spector's notes
        Ehat    =  -2*(Srr*Szz-Srz*Srz)/(alpha);


        #  5 (<-4)
        g       =  -(2*Srz+Szz)*(Srr-Srtheta)/(alpha);

        #  6 (<-5)
        # Note- below could be simplified bc both divided and multiplied by 2
        f1      =  Ehat * (2*Srz+Szz)/2;
        #  6_2 (<-6)
        # Viscoelastic parameters: c, tau 1, tau 2
        f2      = 1 + c*ln( (1+s*tau2)/(1+s*tau1) );



        #  7 (<-8)
        #f      =  r0^2*s / (Ehat*k*f2(c,tau1,tau2))
        # Simplified using tg=r0^2/(Ehat*k)
        # !!Confirm should be a function of c, tau also maybe Sij or tg
        f       = tg * s/f2;


        #  1 (<-8)
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