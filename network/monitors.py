import numpy as np

class MaximalFiringError(Exception): pass
class MinimalFiringError(Exception): pass

class PopulationRateMonitor(object):
    def __init__(self, r_max=1000, r_min=0):
        self.r_max = r_max
        self.r_min = r_min

    def run(self, t, r):
        if np.mean(r) >= self.r_max:
            raise MaximalFiringError()
        if t == 200e-3:
            if np.mean(r) < self.r_min:
                raise MinimalFiringError()
