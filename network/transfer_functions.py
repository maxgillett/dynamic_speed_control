import numpy as np
from numpy import sqrt
from numba import jit, njit

import importlib 
cupy_loader = importlib.find_loader('cupy')
if cupy_loader is not None:
    import cupy as cp
else:
    cp = np


# FIXME: Handle different libpython.so for parallelization with multiple python versions
from optimized_net.routines import Phi as lif_phi_optimized

class RateTransferFunction(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return self.phi(x)

    @property
    def params(self):
        params = dict()
        for param in self.param_list:
            params[param] = getattr(self, param)
        return params

    def set_params(self, param_dict):
        for key, val in param_dict.items():
            setattr(self, key, val)


class LIF(RateTransferFunction):
    def __init__(self, *args, **kwargs):
        """
        Optimized leaky integrate-and-fire transfer function
        """
        super(LIF, self).__init__()
        self.type = 'lif'
        self.param_list = ['type', 'sigma', 'taum', 'thresh', 'reset', 'taurp', 'bounds']
        self.set_params(kwargs)

    # See FIXME not above.
    @property
    def phi(self):
        mu = np.linspace(self.bounds["lower"], self.bounds["upper"], self.bounds["count"])
        r = [lif_phi_optimized(
                x,
                self.sigma,
                self.taum,
                self.thresh,
                self.reset,
                self.taurp) for x in mu]
        def f(x):
            return np.interp(x, mu, r)
        return f


class ReLU(RateTransferFunction):
    def __init__(self, *args, **kwargs):
        """
        Rectified linear unit transfer function
        """
        super(ReLU, self).__init__()
        self.type = 'relu'
        self.param_list = ['type', 'g', 'o', 'phi_max']
        self.default_params = {'o': 0, 'phi_max': np.inf}
        self.set_params(self.default_params)
        self.set_params(kwargs)

    @property
    def phi(self):
        def f(x):
            return (self.g*x + self.o).clip(min=0, max=self.phi_max)
        return f

class ExponentialFunction(RateTransferFunction):
    def __init__(self, *args, **kwargs):
        """
        Exponential transfer function
        """
        super(ExponentialFunction, self).__init__()
        self.type = 'exp'
        self.param_list = ['type', 'g']
        self.default_params = {'g': 1}
        self.set_params(self.default_params)
        self.set_params(kwargs)

    @property
    def phi(self):
        def f(x):
            return np.exp(self.g*x)
        return f


class StepFunction(RateTransferFunction):
    def __init__(self, *args, **kwargs):
        """
        Cumulative density transfer function
        """
        super(ErrorFunction, self).__init__()
        self.type = 'erf'
        self.param_list = ['type', 'mu', 'r_max']
        self.default_params = {'r_max': 1}
        self.set_params(self.default_params)
        self.set_params(kwargs)

    @property
    def phi(self):
        def f(x):
            y = np.zeros_like(x)
            y[x > self.mu] = 1 # r_max
            return y
        return f

    @property
    def phi_(self):
        def f(x):
            y = np.zeros_like(x)
            y[x > self.mu] = 1 # r_max
            return y
        return f

class ErrorFunction(RateTransferFunction):
    def __init__(self, *args, **kwargs):
        """
        Cumulative density transfer function
        """
        super(ErrorFunction, self).__init__()
        self.type = 'erf'
        self.device = kwargs.get('device', 'cpu')
        self.param_list = ['type', 'mu', 'sigma', 'r_max']
        self.default_params = {'r_max': 1}
        self.set_params(self.default_params)
        self.set_params(kwargs)

    @property
    def phi(self):
        if self.device == "cpu":
            def f(x):
                return 0.5 * (1 + erf((x - self.mu) / (sqrt(2) * self.sigma)))
            return f
        elif self.device == "gpu":
            def f(x):
                return 0.5 * (1 + erf_gpu((x - self.mu) / (sqrt(2) * self.sigma)))
            return f

@njit
def erf(x):
    return 1 - erfc(x)

@njit
def erfc(x):
    z=np.abs(x)
    t=1.0/(1.0+0.5*z)
    ans=t*np.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+ \
        t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+ \
        t*(-0.82215223+t*0.17087277)))))))))
    return np.where(x > 0, ans, 2.0-ans)

def erf_gpu(x):
    z=abs(x)
    t=1.0/(1.0+0.5*z)
    ans=t*cp.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+ \
        t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+ \
        t*(-0.82215223+t*0.17087277)))))))))
    return 1 - cp.where(x > 0, ans, 2.0-ans)