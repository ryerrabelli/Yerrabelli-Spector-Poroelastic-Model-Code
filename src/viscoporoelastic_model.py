
import numpy as np
import mpmath
import scipy.optimize
import scipy as sp
import abc

# Below import allows you to annotate the output of functions
# Unlike basic types i.e. dict and list, the object types i.e. Dict and List allow you to do more complex annotations
# like defining the output as Tuple[Union[tuple, Any]] as opposed to just tuple
from typing import Tuple, Union, Any, Dict, List, AnyStr
from collections.abc import Callable
Name = str   # Resource on type hints: https://docs.python.org/3/library/typing.html
Names = Tuple[str]
Value = Any
Values = Tuple[Any]

# Numpy besseli (i0) function doesn't support complex values and only has order 0


"""def I0(x): return sp.special.iv(0, x)  # return np.i0(x); #besseli(0, x)
def I1(x): return sp.special.iv(1, x)  # besseli(1, x)
def J0(x): return sp.special.jv(0, x)
def J1(x): return sp.special.jv(1, x)
def ln(x): return np.log(x)  # import math #return math.log(x)"""



### Start combined matlab/python code


"""
## Fitted parameters (to be determined by experimental fitting to
# the unknown material)
c = 1;
tau1 = 1;
tau2 = 1;
#tau = [tau1, tau2];
#tau = [1 1];
time_const=40.62; #in units of s   # for porosity_sp == 0.5
Vrtheta = 1; # Not actually v, but greek nu (represents Poisson's ratio)
Err = 1;
"""

from numpy import zeros
from numpy import pi

class LibraryEquations(abc.ABC):
    @classmethod
    @abc.abstractmethod  # classmethod is outside of abc.abstractmethod
    def get_core_equations(cls):
        raise NotImplementedError()


class MpmathEquations:
    I0 = np.frompyfunc(lambda x: mpmath.besseli(0,x), nin=1, nout=1)
    I1 = np.frompyfunc(lambda x: mpmath.besseli(1,x), nin=1, nout=1)
    Iv = np.frompyfunc(lambda x,v: mpmath.besseli(v,x), nin=2, nout=1)
    J0 = np.frompyfunc(lambda x: mpmath.besselj(0,x), nin=1, nout=1)
    J1 = np.frompyfunc(lambda x: mpmath.besselj(1,x), nin=1, nout=1)
    Jv = np.frompyfunc(lambda x,v: mpmath.besselj(v,x), nin=2, nout=1)
    ln = np.frompyfunc(mpmath.ln, nin=1, nout=1)
    exp = np.frompyfunc(mpmath.exp, nin=1, nout=1)
    sqrt = np.frompyfunc(mpmath.sqrt, nin=1, nout=1)
    fsolve = np.frompyfunc(mpmath.findroot, nin=2, nout=1)

    @classmethod
    def get_core_equations(cls):
        return cls.I0, cls.I1, cls.J0, cls.J1, cls.ln, cls.exp, cls.sqrt


class ScipyEquations:
    @staticmethod
    def I0(x): return sp.special.iv(0, x)  # return np.i0(x); #besseli(0, x)
    @staticmethod
    def I1(x): return sp.special.iv(1, x)  # besseli(1, x)
    @staticmethod
    def Iv(x, v):    return sp.special.iv(v, x)
    @staticmethod
    def J0(x): return sp.special.jv(0, x)
    @staticmethod
    def J1(x): return sp.special.jv(1, x)
    @staticmethod
    def Jv(x, v):    return sp.special.jv(v, x)
    @staticmethod
    def ln(x): return np.log(x)  # import math #return math.log(x)
    from numpy import exp
    from numpy import sqrt
    from scipy.optimize import fsolve

    @classmethod
    def get_core_equations(cls):
        """Use like I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()"""
        return cls.I0, cls.I1, cls.J0, cls.J1, cls.ln, cls.exp, cls.sqrt


class LaplaceModel(abc.ABC):  # inheriting from abc.ABC means that this is abstract base class

    def __init__(self, equation_library=None):
        self.set_equation_library(equation_library)
        super().__init__()
        #super().__init__(self)

    def set_equation_library(self,equation_library="scipy"):
        if equation_library is not None and equation_library.lower() in ["mpmath","m"]:
            self._equation_library = MpmathEquations
        else:
            self._equation_library = ScipyEquations

    def get_core_equations(self):
        return self._equation_library.get_core_equations()

    @classmethod
    def get_predefined_constants(cls) -> Values:
        return ()  # zero-length tuple, aka tuple()

    @staticmethod
    # This is a static method as predefined_constants is independent of the specific instance
    def get_predefined_constant_names() -> Names:
        return ()  # zero-length tuple, aka tuple()

    # Not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fittable_parameters(self) -> Values:
        return ()  # zero-length tuple, aka tuple()

    @staticmethod
    def get_fittable_parameter_names() -> Names:
        return ()  # zero-length tuple, aka tuple()

    # Not a static method as calculable constants depend on the instance (note- the names are still same though)
    def get_calculable_constants(self) -> Values:
        return ()  # zero-length tuple, aka tuple()

    @staticmethod
    def get_calculable_constant_names() -> Names:
        return ()  # zero-length tuple, aka tuple()

    @classmethod
    def get_var_categories(cls,
                           include_predefined_constants=True,
                           include_fittable_parameters=True,
                           include_calculable_constants=True) -> Tuple[str]:
        # Multiplication operator acts to perform concatenation here
        # Using tuples of strings (alternatively, could use lists instead)
        return sum(  # the sum(.) function acts here to concatenate tuples
            [
                ("Constant",) * len(cls.get_predefined_constant_names()
                                    ) if include_predefined_constants else tuple(),
                ("FittedParam",) * len(cls.get_fittable_parameter_names()
                                       ) if include_fittable_parameters else tuple(),
                ("Calculated",) * len(cls.get_calculable_constant_names()
                                      ) if include_calculable_constants else tuple(),
            ],
            tuple()  # "start" explicitly defined as empty tuple (default is int 0, which throws an error when added
            # with tuples)
        )

    @classmethod
    def get_var_names(cls,
                      include_predefined_constants=True,
                      include_fittable_parameters=True,
                      include_calculable_constants=True) -> Names:
        return sum(  # the sum(.) function acts here to concatenate tuples
            [
                cls.get_predefined_constant_names(
                ) if include_predefined_constants else tuple(),  # 0-length tuple
                cls.get_fittable_parameter_names() if include_fittable_parameters else tuple(),
                cls.get_calculable_constant_names() if include_calculable_constants else tuple(),
            ],
            tuple()  # "start" explicitly defined as empty tuple (default is int 0, which throws an error
            # when added with  tuples)
        )

    def get_var_values(self,
                       include_predefined_constants=True,
                       include_fittable_parameters=True,
                       include_calculable_constants=True) -> Values:
        return sum(  # the sum(.) function acts here to concatenate tuples
            [
                self.get_predefined_constants(
                ) if include_predefined_constants else tuple(),  # 0-length tuple
                self.get_fittable_parameters() if include_fittable_parameters else tuple(),
                self.get_calculable_constants() if include_calculable_constants else tuple(),
            ],
            tuple()  # "start" explicitly defined as empty tuple (default is int 0, which throws an error
            # when added with  tuples)
        )

    # Formerly called get_all_names_and_vars(.)
    def get_var_dict(self, **kwargs) -> Dict[Name, Value]:
        #dict(zip(type(self).get_predefined_constant_names(), self.get_predefined_constants()))
        #dict(zip(type(self).get_fittable_parameter_names(), self.get_fittable_parameters()))
        #dict(zip(type(self).get_calculable_constant_names(), self.get_calculable_constants()))
        """return dict(zip(
            type(self).get_predefined_constant_names() + type(self).get_fittable_parameter_names() + type(
                self).get_calculable_constant_names(),
            self.get_predefined_constants() + self.get_fittable_parameters() + self.get_calculable_constants()
        ))"""

        return {
            name: value for name, value in zip(self.get_var_names(**kwargs), self.get_var_values(**kwargs))
        }

    def get_var_df(self, **kwargs):
        # requires pandas module
        import pandas as pd
        df3 = pd.DataFrame({
            "Value": self.get_var_values(**kwargs),
            "Category": self.get_var_categories(**kwargs)
        },
            index=self.get_var_names(**kwargs)
        )
        return df3

    @abc.abstractmethod
    def laplace_value(self, s):
        raise NotImplementedError()

    @classmethod
    def inverted_value_units(cls) -> str:
        # `return NotImplemented` doesn't throw an error unlike `raise NotImplementedError()`
        return NotImplemented

    @classmethod
    def get_model_name(cls) -> str:
        # self.__class__.__name__
        # type(self).__name__
        return cls.__name__


# inheriting from abc.ABC means that this is abstract base class
class AnalyticallyInvertableModel(LaplaceModel, abc.ABC):
    def inverted_value(self, t): return NotImplemented


# inheriting from abc.ABC means that this is abstract base class
class FittableLaplaceModel(LaplaceModel, abc.ABC):
    pass


class TestModel1(LaplaceModel):
    alpha = 0.5
    tg = 7e-3
    strain_rate = 1e-4
    t0 = 1e3

    @classmethod
    def get_predefined_constants(cls):
        return cls.alpha, cls.tg, cls.strain_rate, cls.t0

    @staticmethod
    def get_predefined_constant_names():
        return "alpha", "tg", "strain_rate", "t0"

    # , s, alpha=0.5, tg=7e-3, strain_rate=1e-4, t0=1e3
    def laplace_value(self, s, alpha=None, tg=None, strain_rate=None, t0=None):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        if alpha is None:
            alpha = self.alpha
        if tg is None:
            tg = self.tg
        if strain_rate is None:
            strain_rate = self.strain_rate
        if t0 is None:
            t0 = self.t0

        eps = tg*strain_rate*(1 - exp(-s*t0/tg))/(s*s)
        F = eps * (3*I0(sqrt(s))-8*alpha*I1(sqrt(s))/sqrt(s)) / \
            (I0(sqrt(s))-2*alpha*I1(sqrt(s))/sqrt(s))
        return F


class TestModel2(AnalyticallyInvertableModel, FittableLaplaceModel):
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
        super().__init__(self)
        self.vs = 0
        self.tg = 7e3  # sec
        self.Es = 7e6  # Pa
        self.eps0 = 0.001
        self.a = 0.003  # meters

        self.alpha2_vals = None
        self.A_vals = None
        self.saved_bessel_len = 0

    def get_fittable_parameters(self):
        return self.vs, self.tg, self.Es, self.eps0, self.a

    @classmethod
    def get_fittable_parameter_names(cls):
        return "vs", "tg", "Es", "eps0", "a"

    def get_calculable_constants(self):
        # type(self).get_predefined_constants()
        vs, tg, Es, eps0, a = self.get_fittable_parameters()
        alpha = (1-2*vs)/(2*(1+vs))
        return alpha,  # 1-length tuple

    @staticmethod
    def get_calculable_constant_names():
        return "alpha",  # 1-length tuple

    def characteristic_eqn(self, x):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        vs, tg, Es, eps0, a = self.get_fittable_parameters()
        return J1(x) - (1 - vs) / (1 - 2 * vs) * x * J0(x)

    def setup_constants(self, bessel_len=20):
        vs, tg, Es, eps0, a = self.get_fittable_parameters()
        alpha2_vals = zeros(shape=bessel_len)
        for n in range(bessel_len):
            # Use (n+1)*pi instead of n*pi bc python is zero-indexed unlike Matlab
            alpha = scipy.optimize.fsolve(
                func=self.characteristic_eqn, x0=(n + 1) * pi)
            alpha2_vals[n] = alpha ** 2

        A_vals = zeros(shape=bessel_len)
        for n in range(bessel_len):
            temp = 1 - 2 * vs
            A_vals[n] = (1 - vs) * temp / (1 + vs) * 1 / \
                (temp * temp * alpha2_vals[n] - temp)

        self.alpha2_vals = alpha2_vals
        self.A_vals = A_vals
        self.saved_bessel_len = bessel_len

    def laplace_value(self, s):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        # type(self).get_predefined_constants()
        vs, tg, Es, eps0, a = self.get_fittable_parameters()
        alpha, = self.get_calculable_constants()

        eps = -eps0/s
        F = eps * (3*I0(sqrt(s))-8*alpha*I1(sqrt(s))/sqrt(s)) / \
            (I0(sqrt(s))-2*alpha*I1(sqrt(s))/sqrt(s))
        return F

    def inverted_value(self, t, bessel_len=20):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        vs, tg, Es, eps0, a = self.get_fittable_parameters()

        if bessel_len > self.saved_bessel_len:
            self.setup_constants(bessel_len=bessel_len)
        A_vals = self.A_vals
        alpha2_vals = self.alpha2_vals

        summation = 0
        for n in range(bessel_len):
            summation += A_vals[n] * exp(-alpha2_vals[n]*t/tg)

        F = pi * a*a * -Es * eps0 * (1 + summation)
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
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        vs, tg, Es, eps0, a = self.get_fittable_parameters(
        )  # TestModel3.get_predefined_constants()
        alpha, = self.get_calculable_constants()

        eps = -eps0/s
        U_a = -eps/2 * (I0(sqrt(s))-4*alpha*I1(sqrt(s))/sqrt(s)) / \
            (I0(sqrt(s))-2*alpha*I1(sqrt(s))/sqrt(s))
        return U_a

    def inverted_value(self, t, bessel_len=20):
        """
        Overrides super function
        :return:
        """
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        vs, tg, Es, eps0, a = self.get_fittable_parameters(
        )  # type(self).get_predefined_constants()

        if bessel_len > self.saved_bessel_len:
            self.setup_constants(bessel_len=bessel_len)
        A_vals = self.A_vals
        alpha2_vals = self.alpha2_vals

        summation_a = 0
        for n in range(bessel_len):
            summation_a += exp(-alpha2_vals[n]*t/tg)/(alpha2_vals[n]-1)

        return summation_a

    @classmethod
    def inverted_value_units(cls):
        return "unitless"  # displacement/a is m/m = unitless


class TestModel4(FittableLaplaceModel):   # Dr. Spector sent this to me May 29, 2021
    """
    v = 0
    strain_rate = 0.0003  #1e-3  # s^-1
    t0_tg = 0.1
    tg = 1000  #7e3  # sec
    """

    def __init__(self):
        super().__init__(self)
        self.v = 0
        self.strain_rate = 0.0003  # 1e-3  # s^-1
        self.t0_tg = 0.1
        self.tg = 1000  # 7e3  # sec

    def get_fittable_parameters(self):
        return self.v, self.strain_rate, self.t0_tg, self.tg

    @classmethod
    def get_fittable_parameter_names(cls):
        return "v", "strain_rate", "t0/tg", "tg"

    def get_calculable_constants(self):
        v, strain_rate, t0_tg, tg = self.get_fittable_parameters()
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
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        v, strain_rate, t0_tg, tg = self.get_fittable_parameters()
        t0, eps0, C0 = self.get_calculable_constants()

        # TODO: Confirm the TestModel4 epszz expression from Dr. Spector as this seems to be different from the one
        #  for the viscoporoelastic model

        # Laplace transform of the axial strain
        epszz = eps0*(1 - exp(-s*t0/tg))/(s*s)
        f_prime = epszz * (3*I0(sqrt(s))-4*C0*I1(sqrt(s))
                           / sqrt(s)) / (I0(sqrt(s))-C0*I1(sqrt(s))/sqrt(s))
        return f_prime


class ViscoporoelasticModel0(FittableLaplaceModel):
    ## PARAMETERS
    ## Predefined constants
    eps0 = 0.1  # 10 percent
    strain_rate = 0.1  # 1 percent per s (normally 1#/s)
    ## Below are directly determined by the mesh deformation part of the
    ## experiment (see our paper with Daniel).  -Dr. Spector
    Vrz = 0.5  # Not actually v, but greek nu (represents Poisson's ratio)
    Ezz = 10  # Note- don't mix up Ezz with epszz

    def __init__(self):
        super().__init__(self)
        self.c = 1
        self.tau1 = 1
        self.tau2 = 1
        # tau = [tau1, tau2];
        # tau = [1 1];
        self.tg = 40.62  # in units of s   # for porosity_sp == 0.5
        # Not actually v, but greek nu (represents Poisson's ratio)
        self.Vrtheta = 1
        self.Err = 1

    @classmethod
    def get_predefined_constants(cls):
        return cls.eps0, cls.strain_rate, cls.Vrz, cls.Ezz

    @staticmethod
    def get_predefined_constant_names():
        return "eps0", "strain_rate", "Vrz", "Ezz"

    # This is not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fittable_parameters(self):
        return self.c, self.tau1, self.tau2, self.tg, self.Vrtheta, self.Err

    @staticmethod
    def get_fittable_parameter_names():
        return "c", "tau1", "tau2", "time_const", "Vrtheta", "Err"

    def get_calculable_constants(self) -> tuple:
        t0 = self.eps0/self.strain_rate
        return (t0,)    # returns tuple of length 1

    @staticmethod
    def get_calculable_constant_names() -> tuple:
        return ("t0",)  # returns tuple of length 1

    def set_fitted_parameters(self,
                              ## Fitted parameters (to be determined by experimental fitting to
                              # the unknown material)
                              c=None,
                              tau1=None,
                              tau2=None,  # tau = [tau1, tau2];
                              tg=None,  # in units of s   # for porosity_sp == 0.5
                              # Not actually v, but greek nu (represents Poisson's ratio)
                              Vrtheta=None,
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
        return self.get_fittable_parameters()

    def laplace_value(self,
                      s,
                      ## Fitted parameters (to be determined by experimental fitting to
                      # the unknown material)
                      c=None,
                      tau1=None,
                      tau2=None,  # tau = [tau1, tau2];
                      tg=None,  # in units of s   # for porosity_sp == 0.5
                      # Not actually v, but greek nu (represents Poisson's ratio)
                      Vrtheta=None,
                      Err=None,
                      ):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        c, tau1, tau2, tg, Vrtheta, Err = self.set_fitted_parameters(
            c=c, tau1=tau1, tau2=tau2, tg=tg, Vrtheta=Vrtheta, Err=Err)

        eps0, strain_rate, Vrz, Ezz = self.get_predefined_constants()

        #print(s)
        ## BASE EQUATIONS
        #  1
        #eps0 = strain_rate * t0
        t0 = eps0/strain_rate
        # TODO: Confirm the TestModel4 epszz expression from Dr. Spector as this seems to be different from the one
        #  for the viscoporoelastic model
        epszz = 1 - exp(-s*t0)/(s*s)  # Laplace transform of the axial strain

        #  2
        Srr = 1/Err
        Srtheta = -Vrtheta/Err
        Srz = -Vrz/Err
        Szz = 1/Ezz
        #Sij     = [Srr, Srtheta, Srz;   Srtheta, Srr, Srz;   Srz, Srz, Szz];

        #  3
        alpha = 2*Srz*Srz-Szz*Srtheta-Srr*Szz
        C13 = Srz/(alpha)
        C33 = -(Srr+Srtheta)/(alpha)

        #  4
        g = -(2*Srz+Szz)*(Srr-Srtheta)/(alpha)

        #  5
        # Note- below could be simplified bc both divided and multiplied by 2
        f1 = -(2*Srz+Szz)/2 * 2*(Srr*Szz-Srz*Srz)/(alpha)

        #  6
        # Viscoelastic parameters: c, tau 1, tau 2
        f2 = 1 + c*ln((1+s*tau2)/(1+s*tau1))

        #  7
        # Note- Ehat is a function of Sij although wasn't stated in Spector's notes
        Ehat = -2*(Srr*Szz-Srz*Srz)/(alpha)

        #  8
        #f      =  r0^2*s / (Ehat*k*f2(c,tau1,tau2))
        # Simplified using time_const=r0^2/(Ehat*k)
        # !!Confirm should be a function of c, tau also maybe Sij or time_const
        f = tg * s/f2

        sigbar =  \
            2*epszz*(
                C13
                * (
                        g
                        * I1(sqrt(f))/sqrt(f)
                        / (Ehat*I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f))
                        - 1/2
                    )
                + C33/2
                + f1
                * f2
                * (I0(sqrt(f))-2*I1(sqrt(f))/sqrt(f))
                / (2 * (Ehat*I0(sqrt(f)) - I1(sqrt(f))/sqrt(f)))
            )

        return sigbar


class ArmstrongIsotropicModel(FittableLaplaceModel):   # Aka TestModel5
    # Dr. Spector sent this model to me May 29, 2021, then revised it on Jun 11, 2021
    """
    v = 0
    strain_rate = 0.0003  #1e-3  # s^-1
    t0_tg = 0.1
    tg = 1000  #7e3  # sec
    """

    def __init__(self):
        super().__init__(self)
        self.v = 0
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

    def get_fittable_parameters(self):
        return self.v, self.strain_rate, self.t0_tg, self.tg

    @classmethod
    def get_fittable_parameter_names(cls):
        return "v", "strain_rate", "t0/tg", "tg"

    def get_calculable_constants(self):
        v, strain_rate, t0_tg, tg = self.get_fittable_parameters()
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
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        v, strain_rate, t0_tg, tg = self.get_fittable_parameters()
        t0, eps0, C0 = self.get_calculable_constants()

        # TODO: Confirm the TestModel4 epszz expression from Dr. Spector as this seems to be different from the one
        #  for the viscoporoelastic model

        # Laplace transform of the axial strain
        epszz = strain_rate*tg*(1 - exp(-s*t0/tg))/(s*s)
        f_prime = epszz * (3*I0(sqrt(s))-4*C0*I1(sqrt(s))
                           / sqrt(s)) / (I0(sqrt(s))-C0*I1(sqrt(s))/sqrt(s))
        return f_prime


class ViscoporoelasticModel1(FittableLaplaceModel):
    ## PARAMETERS
    ## Predefined constants
    t0_tg = 0.1
    strain_rate = 0.1  # 1 percent per s (normally 1#/s)
    ## Below are directly determined by the mesh deformation part of the
    ## experiment (see our paper with Daniel).  -Dr. Spector
    Vrz = 0.5  # Not actually v, but greek nu (represents Poisson's ratio)
    Ezz = 10  # Note- don't mix up Ezz with epszz

    def __init__(self, c=1, tau1=1, tau2=1, tg=40.62, Vrtheta=1, Err=1):
        """
        self.c = 1;
        self.tau1 = 1;
        self.tau2 = 1;
        # tau = [tau1, tau2];
        # tau = [1 1];
        self.tg = 40.62;  # in units of s   # for porosity_sp == 0.5
        self.Vrtheta = 1;  # Not actually v, but greek nu (represents Poisson's ratio)
        self.Err = 1;
        """
        # Source: https://stackoverflow.com/questions/12191075/is-there-a-shortcut-for-self-somevariable-somevariable-in-a-python-class-con/12191118
        vars(self).update((k, v) for k, v in vars().items() if k != "self")
        super().__init__(self)

    @classmethod
    def get_predefined_constants(cls):
        return cls.t0_tg, cls.strain_rate, cls.Vrz, cls.Ezz
        #return type(self).eps0, type(self).strain_rate, type(self).Vrz, type(self).Ezz

    @staticmethod
    def get_predefined_constant_names():
        return "t0/tg", "strain_rate", "Vrz", "Ezz"

    # This is not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fittable_parameters(self):
        return self.c, self.tau1, self.tau2, self.tg, self.Vrtheta, self.Err

    @staticmethod
    def get_fittable_parameter_names():
        return "c", "tau1", "tau2", "tg", "Vrtheta", "Err"

    def get_calculable_constants(self) -> tuple:
        t0 = self.t0_tg * self.tg
        return (t0,)    # returns tuple of length 1

    @staticmethod
    def get_calculable_constant_names() -> tuple:
        return ("t0",)  # returns tuple of length 1

    def set_fitted_parameters(self,
                              ## Fitted parameters (to be determined by experimental fitting to
                              # the unknown material)
                              c=None,
                              tau1=None,
                              tau2=None,  # tau = [tau1, tau2];
                              tg=None,  # in units of s   # for porosity_sp == 0.5
                              # Not actually v, but greek nu (represents Poisson's ratio)
                              Vrtheta=None,
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
        return self.get_fittable_parameters()

    def laplace_value(self,
                      s,
                      ## Fitted parameters (to be determined by experimental fitting to
                      # the unknown material)
                      c=None,
                      tau1=None,
                      tau2=None,  # tau = [tau1, tau2];
                      tg=None,  # in units of s   # for porosity_sp == 0.5
                      # Not actually v, but greek nu (represents Poisson's ratio)
                      Vrtheta=None,
                      Err=None,
                      ):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        c, tau1, tau2, tg, Vrtheta, Err = self.set_fitted_parameters(
            c=c, tau1=tau1, tau2=tau2, tg=tg, Vrtheta=Vrtheta, Err=Err)

        t0_tg, strain_rate, Vrz, Ezz = self.get_predefined_constants()

        #print(s)
        ## BASE EQUATIONS
        #  2 (<-1)
        # Below lines modified from March to June 2021 versions
        t0 = t0_tg * tg
        #eps0 = strain_rate * t0
        # Laplace transform of the axial strain
        epszz = strain_rate * tg * (1 - exp(-s*t0/tg))/(s*s)

        #  3 (<-2)
        Srr = 1/Err
        Srtheta = -Vrtheta/Err
        Srz = -Vrz/Err
        Szz = 1/Ezz
        #Sij     = [Srr, Srtheta, Srz;   Srtheta, Srr, Srz;   Srz, Srz, Szz];

        #  4 (<-3)
        alpha = 2*Srz*Srz-Szz*Srtheta-Srr*Szz
        C13 = Srz/(alpha)
        C33 = -(Srr+Srtheta)/(alpha)
        # Note- Ehat is a function of Sij although wasn't stated in Spector's notes
        Ehat = -2*(Srr*Szz-Srz*Srz)/(alpha)

        #  5 (<-4)
        g = -(2*Srz+Szz)*(Srr-Srtheta)/(alpha)

        #  6 (<-5)
        # Note- below could be simplified bc both divided and multiplied by 2
        f1 = -Ehat * (2*Srz+Szz)/2
        #  6_2 (<-6)
        # Viscoelastic parameters: c, tau 1, tau 2
        f2 = 1 + c*ln((1+s*tau2)/(1+s*tau1))

        #  7 (<-8)
        #f      =  r0^2*s / (Ehat*k*f2(c,tau1,tau2))
        # Simplified using tg=r0^2/(Ehat*k)
        # !!Confirm should be a function of c, tau also maybe Sij or tg
        f = tg * s/f2

        #  1 (<-8)
        I1rtf = I1(sqrt(f))
        # np.isinf returns true if element is inf or -inf (does this elementwise for array)
        is_inf = np.any(np.isinf(I1rtf), axis=-1)
        # I1rtf overall is of type numpy.ndarray, but each value is possibly complex128, which is different than the default complex
        # a complex128 inf throws a warning when dividing by another value, whereas a regular complex doesn't
        is_complex128 = [
            type(I1rtf_val) == np.complex128 for I1rtf_val in I1rtf[is_inf]]
        if np.any(is_inf):
            print(f"Warning the function could not be inverted at some values of t as the I1(sqrt(f)) component "
                  f"led to +/- infinity. The indices of these time points are {np.nonzero(is_inf)}.")
            #I1rtf[np.isinf(I1rtf)] = np.nan

        with np.errstate(invalid="ignore"):
            I1rtf_f = I1rtf / sqrt(f)
            """
            sigbar = \
                2 * epszz * ( \
                            C13 \
                            * ( \
                                        g * I1rtf_f \
                                        / (Ehat * I0(sqrt(f)) - 2 * I1rtf_f) \
                                        - 1 / 2 \
                                ) \
                            + C33 / 2 \
                            + f1 * f2 * \
                            (I0(sqrt(f)) - 2 * I1rtf_f) \
                            / (2 * Ehat * I0(sqrt(f) - 2 * I1rtf_f)) \
                    );
            """
            sigbar = \
                2 * epszz * (
                            C13
                            * (
                                        g
                                        * I1rtf_f
                                        / (Ehat * I0(sqrt(f)) - 2 * I1rtf_f)
                                        - 1 / 2
                                )
                            + C33 / 2
                            + f1
                            * f2
                            * (I0(sqrt(f)) - 2 * I1rtf_f)
                            / (2 * (Ehat * I0(sqrt(f)) - I1rtf_f))
                    )

        return sigbar


class ViscoporoelasticModel2(FittableLaplaceModel):

    def __init__(self, c=2, tau1=0.001, tau2=10, tg=40.62, v=0.3, t0_tg=10):
        """
        self.c = 2;
        self.tau1 = 0.001;
        self.tau2 = 10;
        # tau = [tau1, tau2];
        # tau = [1 1];
        self.tg = 40.62;  # in units of s   # for porosity_sp == 0.5
        self.v = 0.3
        self.t0_tg = 10
        """
        # Source: https://stackoverflow.com/questions/12191075/is-there-a-shortcut-for-self-somevariable-somevariable-in-a-python-class-con/12191118
        vars(self).update((k, v) for k, v in vars().items() if k != "self")
        super().__init__(self)

    @classmethod
    def get_predefined_constants(cls):
        return tuple()

    @staticmethod
    def get_predefined_constant_names():
        return tuple()

    # This is not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fittable_parameters(self):
        return self.c, self.tau1, self.tau2, self.tg, self.v, self.t0_tg

    @staticmethod
    def get_fittable_parameter_names():
        return "c", "tau1", "tau2", "tg", "v", "t0/tg"

    def get_calculable_constants(self) -> tuple:
        t0 = self.t0_tg * self.tg
        return (t0,)    # returns tuple of length 1

    @staticmethod
    def get_calculable_constant_names() -> tuple:
        return ("t0",)  # returns tuple of length 1

    def set_fitted_parameters(self,
                              ## Fitted parameters (to be determined by experimental fitting to
                              # the unknown material)
                              c=None,
                              tau1=None,
                              tau2=None,  # tau = [tau1, tau2];
                              tg=None,  # in units of s   # for porosity_sp == 0.5
                              # Not actually v, but greek nu (represents Poisson's ratio)
                              v=None,
                              t0_tg=None,
                              ):
        if c is not None:
            self.c = c
        if tau1 is not None:
            self.tau1 = tau1
        if tau2 is not None:
            self.tau2 = tau2
        if tg is not None:
            self.tg = tg
        if v is not None:
            self.v = v
        if t0_tg is not None:
            self.t0_tg = t0_tg
        return self.get_fittable_parameters()

    def laplace_value(self,
                      s,
                      ## Fitted parameters (to be determined by experimental fitting to
                      # the unknown material)
                      c=None,
                      tau1=None,
                      tau2=None,  # tau = [tau1, tau2];
                      tg=None,  # in units of s   # for porosity_sp == 0.5
                      # Not actually v, but greek nu (represents Poisson's ratio)
                      v=None,
                      t0_tg=None,
                      ):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        c, tau1, tau2, tg, v, t0_tg = self.set_fitted_parameters(
            c=c, tau1=tau1, tau2=tau2, tg=tg, v=v, t0_tg=t0_tg)

        _ = self.get_predefined_constants()

        gamma = 2*(1-2*v)/(3*(1-v))
        f = s/(1+gamma*c*ln((1+s*tau2)/(1+s*tau1)))
        #T_bar = tg/t0 * coth(sqrt(f))/(s*sqrt(f))*(1-exp(-t0_tg*s))
        T_bar = (1 - exp(-t0_tg * s)) / \
            (t0_tg * s * sqrt(f) * np.tanh(sqrt(f)))

        return T_bar


class ViscoporoelasticModel3(FittableLaplaceModel):
    ## PARAMETERS
    ## Predefined constants
    t0_tg = 10/40.62
    strain_rate = 0.1  # 1 percent per s (normally 1#/s)
    ## Below are directly determined by the mesh deformation part of the
    ## experiment (see our paper with Daniel).  -Dr. Spector
    Vrz = 0.5  # Not actually v, but greek nu (represents Poisson's ratio)
    Ezz = 10  # Note- don't mix up Ezz with epszz

    def __init__(self, c=1, tau1=1, tau2=1, tg=40.62, Vrtheta=1, Err=1):
        """
        self.c = 1;
        self.tau1 = 1;
        self.tau2 = 1;
        # tau = [tau1, tau2];
        # tau = [1 1];
        self.tg = 40.62;  # in units of s   # for porosity_sp == 0.5
        self.Vrtheta = 1;  # Not actually v, but greek nu (represents Poisson's ratio)
        self.Err = 1;
        """
        # Source: https://stackoverflow.com/questions/12191075/is-there-a-shortcut-for-self-somevariable-somevariable-in-a-python-class-con/12191118
        vars(self).update((k, v) for k, v in vars().items() if k != "self")
        super().__init__(self)

    @classmethod
    def get_predefined_constants(cls):
        return cls.t0_tg, cls.strain_rate, cls.Vrz, cls.Ezz
        #return type(self).eps0, type(self).strain_rate, type(self).Vrz, type(self).Ezz

    @staticmethod
    def get_predefined_constant_names():
        return "t0/tg", "strain_rate", "Vrz", "Ezz"

    # This is not a static method as fitted parameters depend on the instance (note- the names are still same though)
    def get_fittable_parameters(self):
        return self.c, self.tau1, self.tau2, self.tg, self.Vrtheta, self.Err

    @staticmethod
    def get_fittable_parameter_names():
        return "c", "tau1", "tau2", "tg", "Vrtheta", "Err"

    def get_calculable_constants(self) -> tuple:
        t0 = self.t0_tg * self.tg
        return (t0,)    # returns tuple of length 1

    @staticmethod
    def get_calculable_constant_names() -> tuple:
        return ("t0",)  # returns tuple of length 1

    def set_fitted_parameters(self,
                              ## Fitted parameters (to be determined by experimental fitting to
                              # the unknown material)
                              c=None,
                              tau1=None,
                              tau2=None,  # tau = [tau1, tau2];
                              tg=None,  # in units of s   # for porosity_sp == 0.5
                              # Not actually v, but greek nu (represents Poisson's ratio)
                              Vrtheta=None,
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
        return self.get_fittable_parameters()

    def laplace_value(self,
                      s,
                      ## Fitted parameters (to be determined by experimental fitting to
                      # the unknown material)
                      c=None,
                      tau1=None,
                      tau2=None,  # tau = [tau1, tau2];
                      tg=None,  # in units of s   # for porosity_sp == 0.5
                      # Not actually v, but greek nu (represents Poisson's ratio)
                      Vrtheta=None,
                      Err=None,
                      return_error_inds=False,
                      ):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        c, tau1, tau2, tg, Vrtheta, Err = self.set_fitted_parameters(c=c, tau1=tau1, tau2=tau2, tg=tg, Vrtheta=Vrtheta,
                                                                     Err=Err)

        t0_tg, strain_rate, Vrz, Ezz = self.get_predefined_constants()

        # print(s)
        ## BASE EQUATIONS
        #  2 (<-1)
        # Below lines modified from March to June 2021 versions
        t0 = t0_tg * tg
        # eps0 = strain_rate * t0
        # Laplace transform of the axial strain
        epszz = strain_rate * tg * (1 - exp(-s * t0 / tg)) / (s * s)

        #  3 (<-2)
        Srr = 1 / Err
        Srtheta = -Vrtheta / Err  # Srσ
        Srz = -Vrz / Err
        Szz = 1 / Ezz
        # Sij     = [Srr, Srtheta, Srz;   Srtheta, Srr, Srz;   Srz, Srz, Szz];

        #  4 (<-3)
        #alpha = 2 * Srz * Srz - Szz * Srtheta - Srr * Szz;
        alpha = 2 * (Srz*Srz)*(Szz/Srr) - Srr*Srtheta - Srr*Srr
        beta = 2*(Srz*Srz) * (Szz/Srr*Szz/Srr) - Szz*Srtheta - Srr*Szz
        gamma = 2*(Srz*Srz)-Szz*Srtheta-Srr*Szz
        C13 = Srz / (alpha)
        C33 = -(Srr + Srtheta) / (beta)
        # Note- Ehat is a function of Sij although wasn't stated in Spector's notes
        Ehat = -2 * (Srr * Szz - Srz*Srz) / (gamma)

        #  5 (<-4)
        g1 = -(2 * Srz + Szz) * (Srr - Srtheta) / (gamma)

        #  6 (<-5)
        # Note- below could be simplified bc both divided and multiplied by 2
        f1 = -Ehat * (2*Srz + Szz) / (2*gamma)
        #  6_2 (<-6)
        # Viscoelastic parameters: c, tau 1, tau 2
        f2 = 1 + c * ln((1 + s * tau2) / (1 + s * tau1))

        #  7 (<-8)
        # f      =  r0^2*s / (Ehat*k*f2(c,tau1,tau2))
        # Simplified using tg=r0^2/(Ehat*k)
        # !!Confirm should be a function of c, tau also maybe Sij or tg
        f = tg * s / f2

        #  1 (<-8)
        I1rtf = I1(sqrt(f))
        # np.isinf returns true if element is inf or -inf (does this elementwise for array)
        is_inf = np.any(np.isinf(I1rtf), axis=-1)
        # I1rtf overall is of type numpy.ndarray, but each value is possibly complex128, which is different than the default complex
        # a complex128 inf throws a warning when dividing by another value, whereas a regular complex doesn't
        is_complex128 = [
            type(I1rtf_val) == np.complex128 for I1rtf_val in I1rtf[is_inf]]
        if np.any(is_inf):
            # I1rtf[np.isinf(I1rtf)] = np.nan
            if return_error_inds:
                error_inds = is_inf
            else:
                import utils
                (indices_is_inf, ) = np.nonzero(is_inf)
                print(f"Warning the function could not be inverted at some ({len(indices_is_inf)}/{len(is_inf)}) values of t as the I1(sqrt(f)) component "
                      f"led to +/- infinity. The indices of these time points are {utils.abbreviate(indices_is_inf)}.")
        else:
            error_inds = []

        with np.errstate(invalid="ignore"):
            I1rtf_f = I1rtf / sqrt(f)
            I0rtf = I0(sqrt(f))
            """
            sigbar = \
                2 * epszz * ( \
                            C13 \
                            * ( \
                                        g * I1rtf_f \
                                        / (Ehat * I0(sqrt(f)) - 2 * I1rtf_f) \
                                        - 1 / 2 \
                                ) \
                            + C33 / 2 \
                            + f1 * f2 * \
                            (I0(sqrt(f)) - 2 * I1rtf_f) \
                            / (2 * Ehat * I0(sqrt(f) - 2 * I1rtf_f)) \
                    );
            """
            sigbar = \
                2 * epszz * (
                            C13 * g1 * I1rtf_f
                            / (Ehat * I0rtf - 2 * I1rtf_f)
                            + (C33-C13) / 2
                            + f1 * f2
                            * (I0rtf - I1rtf_f)
                            / (2*Ehat*I0rtf - 4*I1rtf_f)
                    )

        if return_error_inds:
            return (sigbar, error_inds)
        else:
            return sigbar


class CohenModel(LaplaceModel):
    """
    Source of equation:
    Cohen, B., Lai, W. M., and Mow, V. C. (August 1, 1998). "A Transversely Isotropic Biphasic Model for Unconfined
    Compression of Growth Plate and Chondroepiphysis." ASME. J Biomech Eng. August 1998; 120(4): 491–496.
    https://doi.org/10.1115/1.2798019
    """

    t0_tg = 10 / 40.62  # unitless,  t0_tg=None indicates stepwise strain
    tg = 40.62  # sec
    strain_rate = 0.01  # per sec
    E1 = 8.5  # kPa
    E3 = 19   # kPa
    v21 = 0.75  # like Vrtheta
    v31 = 0.24  # like Vrz

    def __init__(self, **kwargs):
        self.alpha2_vals = None
        self.saved_bessel_len = 0

        # Source: https://stackoverflow.com/questions/12191075/is-there-a-shortcut-for-self-somevariable-somevariable-in-a-python-class-con/12191118
        vars(self).update((k, v) for k, v in vars().items() if k != "self")
        super().__init__()

    @classmethod
    def get_predefined_constants(cls):
        return cls.t0_tg, cls.tg, cls.strain_rate, cls.E1, cls.E3, cls.v21, cls.v31

    @staticmethod
    def get_predefined_constant_names():
        return "t0_tg", "tg", "strain_rate", "E1", "E3", "v21", "v31"

    @staticmethod
    def get_predefined_constant_names_latex():
        """The $ is not included inside the returned strings"""
        return "t_0/t_g", "t_g", r"\dot{\varepsilon}",     \
               "E_1", "E_3", r"\nu_{21}", r"\nu_{31}"

    def get_calculable_constants(self):
        t0_tg, tg, strain_rate, E1, E3, v21, v31 = self.get_predefined_constants()
        v31sq = v31 * v31

        delta1 = 1 - v21 - 2*v31sq*E1/E3
        delta2 = (1 - v31sq*E1/E3)/(1+v21)
        delta3 = (1 - 2*v31sq)*delta2/delta1

        C11 = E1*(1 - v31sq * E1/E3) / ((1+v21) * delta1)
        C12 = E1*(v21+v31sq * E1/E3) / ((1+v21) * delta1)
        C13 = E1*v31 / delta1
        C33 = E3 * (1 + 2*v31sq * E1/E3 / delta1)   # C44==C31

        C0 = (C11-C12)/C11
        C1 = (2*C33 + C11 + C12 - 4*C13) / (C11-C12)
        C2 = 2 * (C33*(C11-C12) + C11*(C11+C12-4*C13)
                  + 2*C13*C13) / (C11-C12)**2

        # Units of delta1, delta2, and delta3 are non-dimensional
        # Units of C11, C12, C13, and C33 are pressure units (the same units as E1 and E3 so kPa in this case)
        # Units of C0, C1, and C2 are non-dimensional
        # The factor to make things dimensional is to multiply by (C11-C12)/2
        return delta1, delta2, delta3, C11, C12, C13, C33, C0, C1, C2,

    @staticmethod
    def get_calculable_constant_names():
        return "Δ1", "Δ2", "Δ3", "C11", "C12", "C13", "C33", "C0", "C1", "C2",

    def laplace_value(self, s, dimensional: bool = True, eps_zz: np.ndarray = None):
        """
        Result units are in pressure units (the same units as E1 and E3 so kPa in this case)
        :param s:
        :type s:
        :param dimensional:
        :type dimensional: bool
        :return:
        :rtype:
        """
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        t0_tg, tg, strain_rate, E1, E3, v21, v31 = self.get_predefined_constants()

        delta1, delta2, delta3, C11, C12, C13, C33, C0, C1, C2 = self.get_calculable_constants()

        if eps_zz is None:
            if t0_tg is None:  # Stepwise strain (not ramped)
                # Recall e^x ≈ 1+x for small x
                # lim as t0->0 of (1 - exp(-t0_tg * s)) / (s*s)
                # = (1- (1-t0_tg*s) )/(s*s)
                # = ( t0_tg*s )/(s*s) = t0_tg/s
                eps_zz = strain_rate * tg * 1/s
            else:
                eps_zz = strain_rate * tg * (1 - exp(-t0_tg * s)) / (s*s)

        #I1rts = I1(sqrt(s))
        I1rts_s = I1(sqrt(s)) / sqrt(s)
        I0rts = I0(sqrt(s))

        # F is the load intensity
        F = (C1*I0rts - C2*C0*I1rts_s) / (I0rts - C0*I1rts_s) * eps_zz

        if dimensional:
            F = F * (C11-C12)/2

        return F

    def characteristic_eqn(self, x):
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        t0_tg, tg, strain_rate, E1, E3, v21, v31 = self.get_predefined_constants()
        return J1(x) - (1 - v31*v31*E1/E3) / (1 - v21 - 2*v31*v31*E1/E3) * x * J0(x)

    def setup_constants(self, bessel_len=20):
        alpha2_vals = zeros(shape=bessel_len)
        for n in range(1, bessel_len+1):
            # indexed from 1 to be similar to Matlab code (n is 1,2,...bessel_len)
            alpha = self._equation_library.fsolve(self.characteristic_eqn, n*pi)
            alpha2_vals[n-1] = alpha**2

        self.alpha2_vals = alpha2_vals
        self.saved_bessel_len = bessel_len

    @classmethod
    def inverted_value_units(cls):
        # Result units are in pressure units (the same units as E1 and E3 so kPa in this case)
        return "kPa"

    def inverted_value(self, t, bessel_len=20):
        """
        Implemented (and override) the inherited method
        :param t:
        :type t:
        :param bessel_len:
        :type bessel_len:
        :return:
        :rtype:
        """
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        t0_tg, tg, strain_rate, E1, E3, v21, v31 = self.get_predefined_constants()

        delta1, delta2, delta3, _, _, _, _, C0, C1, C2 = self.get_calculable_constants()

        if bessel_len > self.saved_bessel_len:
            self.setup_constants(bessel_len=bessel_len)
        alpha2_vals = self.alpha2_vals

        #F = zeros(shape=np.array(t).shape )
        if t0_tg is None:  # stepwise strain (not ramped)
            F = E3 * strain_rate + E1 * strain_rate * delta3 * sum(exp(-alpha2_N*t/tg)/( delta2*delta2*alpha2_N - delta1/(1+v21) ) for alpha2_N in alpha2_vals )

        else:
            # Ramped strain
            #F = E3*strain_rate*t + \
            #    E1*strain_rate*tg * (1/8 + sum(exp(-alpha2_N*t/tg)/denom for alpha2_N in alpha2_vals) )
            #F = E3*strain_rate*t - \
            #    E1*strain_rate*tg*delta3 * \
            #    sum((exp(-alpha2_N*t/tg)-exp(-alpha2_N*(t/tg-t0_tg)))/denom for alpha2_N in alpha2_vals)
            #print(f"delta1/(1+v21)={delta1/(1+v21)}")
            #print(f"E1 * strain_rate * tg * delta3={E1 * strain_rate * tg * delta3}")

            """
            F = np.where(
                t / tg < t0_tg,
                E3 * strain_rate * t + \
                E1 * strain_rate * tg * delta3 *
                (1/8 - sum(exp(-alpha2_N * t/tg) / (alpha2_N*(delta2*delta2*alpha2_N - delta1/(1+v21)))
                           for alpha2_N in alpha2_vals
                           )
                 ),
                E3 * strain_rate * t0_tg*tg - \
                E1 * strain_rate * tg * delta3 *
                sum((exp(-alpha2_N * t/tg) - exp(-alpha2_N * (t/tg - t0_tg))) / (alpha2_N*(delta2*delta2*alpha2_N - delta1/(1+v21)))
                    for alpha2_N in alpha2_vals
                    )
                )
            return F
            """

            F = np.piecewise(t,
                             [t/tg < 0,
                             (t/tg >= 0) & (t/tg < t0_tg),
                              t/tg >= t0_tg],
                             [0,
                                 (
                                     lambda t:
                                     E3 * strain_rate * t
                                     + E1 * strain_rate * tg * delta3
                                     * (1/8 - sum(
                                         exp(-alpha2_N * t/tg)
                                         / (alpha2_N*(delta2*delta2*alpha2_N - delta1/(1+v21)))
                                         for alpha2_N in alpha2_vals
                                         )
                                        )
                                     ),
                                 (
                                     lambda t:
                                     E3 * strain_rate * t0_tg*tg
                                     - E1 * strain_rate * tg * delta3
                                     * sum(
                                         (exp(-alpha2_N * t/tg)
                                          - exp(-alpha2_N * (t/tg - t0_tg)))
                                         / (alpha2_N*(delta2*delta2*alpha2_N - delta1/(1+v21)))
                                         for alpha2_N in alpha2_vals
                                         )
                                     )]
                             )

        return F


class CohenModel1998(CohenModel):
    t0_tg = 1
    tg = 1  # sec
    strain_rate = 0.01  # per sec
    E1 = 5  # kPa
    E3 = 1  # kPa
    v21 = 0.3  # like Vrtheta
    v31 = 0  # like Vrz

    def inverted_valuex(self, t, bessel_len=20, dimensional: bool = True):
        """
        Implemented (and override) the inherited method
        Result units are in pressure units (the same units as E1 and E3 so kPa in this case)
        :param t:
        :type t:
        :param bessel_len:
        :type bessel_len: bool
        :return:
        :rtype:
        """
        I0, I1, J0, J1, ln, exp, sqrt = self.get_core_equations()
        t0_tg, tg, strain_rate, E1, E3, v21, v31 = self.get_predefined_constants()
        t0 = t0_tg*tg
        delta1, delta2, delta3, C11, C12, C13, C33, C0, C1, C2 = self.get_calculable_constants()
        E1_E3 = E1/E3
        if bessel_len > self.saved_bessel_len:
            self.setup_constants(bessel_len=bessel_len)
        alpha2_vals = self.alpha2_vals

        part1 = np.minimum(t0_tg, 1)
        summation = []
        for alpha2 in alpha2_vals:
            denom = alpha2 * (delta2**2 * alpha2 - delta1/(1+v21))
            summation.append(exp(-alpha2*tg*t)/denom)
        summation = np.sum(summation, axis=0)
        part2 = E1_E3*delta3*(1/8-summation)
        F = E3*strain_rate*t0 * (part1+part2)

        if dimensional:
            F = F * (C11-C12)/2

        return F


def getCohenModelModified(**kwargs):
    class CohenModelModified(CohenModel):
        # pylint: disable=E221, E272
        # Below comment disables pylint warning about whitespace around =
        # https://pycodestyle.pycqa.org/en/latest/intro.html

        superclass = CohenModel
        t0_tg = kwargs.get("t0_tg", superclass.t0_tg)  # 10 / 40.62; # unitless
        tg    = kwargs.get("tg",       superclass.tg)  # 40.62  # sec
        strain_rate =    \
                kwargs.get("strain_rate", superclass.strain_rate)  # 0.01;  # 1/sec

        # E1, E3 vars in kPa
        E1    = kwargs.get("E1",      superclass.E1)  # 8.5 # kPa
        E3    = kwargs.get("E3",      superclass.E3)  # 19  # kPa

        # v21 and v31 are unitless. Note- "v" actually represents greek nu (ν)
        v21   = kwargs.get("v21",     superclass.v21)  # 0.75 # like Vrtheta, unitless
        v31   = kwargs.get("v31",     superclass.v31)  # 0.24 # like Vrz, unitless

        if kwargs.get("t0") is not None:
            if not kwargs.get("t0_tg"):
                t0_tg = kwargs.get("t0") / tg
            if not kwargs.get("tg"):
                tg = t0_tg / kwargs.get("t0")

        if kwargs.get("E1_E3") is not None:
            if not kwargs.get("E1"):
                E1 = kwargs.get("E1_E3") * E3
            if not kwargs.get("E3"):
                E3 = E1 / kwargs.get("E1_E3")

    return CohenModelModified()


if __name__ == '__main__':
    s_val = 0.001
    vpe = ViscoporoelasticModel1()
    output = s_val * vpe.laplace_value(s_val)
    print(output)
