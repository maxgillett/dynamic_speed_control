import logging
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from helpers import spike_to_rate

import importlib 
cupy_loader = importlib.find_loader('cupy')
if cupy_loader is not None:
    import cupy as cp
else:
    cp = np

logger = logging.getLogger(__name__)

class Sequence(object):
    def __init__(self):
        "P x N"
        self.inputs = None
        self.device = 'cpu'

    def overlaps(self, 
            net,
            pop,
            phi=None,
            plasticity=None,
            spikes=False,
            correlation=False,
            disable_pbar=False):

        if correlation:
            logger.info("Computing correlations")
        else:
            logger.info("Computing overlaps")

        if self.device == "cpu":
            xp = np
        elif self.device == "gpu":
            xp = cp
            if spikes:
                raise NotImplementedError

        overlaps = []
        inputs = self.inputs
        if phi:
            inputs = phi(inputs)
        if plasticity:
            inputs = plasticity.g(inputs)
        for pattern in tqdm(inputs, disable=disable_pbar):
            if correlation:
                if spikes:
                    rate = spike_to_rate(pop.spikes)
                    overlap = xp.asarray(
                        [xp.corrcoef(pattern, rate[:,t])[0,1] for t in range(rate.shape[1])])
                else:
                    if net.formulation == 1:
                        overlap = xp.asarray(
                            [xp.corrcoef(pattern, pop.state[:,t])[0,1] 
                                for t in range(net.exc.state.shape[1])])
                    elif net.formulation == 2:
                        overlap = xp.asarray(
                            [xp.corrcoef(pattern, pop.phi(pop.state[:,t]))[0,1]
                                for t in range(net.exc.state.shape[1])])
                    elif net.formulation == 4:
                        overlap = xp.asarray(
                            [xp.corrcoef(pattern, pop.phi(pop.state[:,t]))[0,1] 
                                for t in range(net.exc.state.shape[1])])
                    elif net.formulation == 5:
                        overlap = xp.asarray(
                            [xp.corrcoef(pattern, pop.phi(pop.state[:,t]))[0,1] 
                                for t in range(net.exc.state.shape[1])])
                    else:
                        raise StandardError("Unknown rate formulation")
            else:
                overlap = net.overlap_with(pattern, pop, spikes)
            overlaps.append(overlap)
        return np.vstack(overlaps)


class GaussianSequence(Sequence):
    def __init__(self, P, N, mu=0, sigma=1, seed=42, device='cpu'):
        super(Sequence, self).__init__()
        self.device = device
        self.inputs = np.random.RandomState(seed).normal(mu,sigma,size=(P,N))
        if self.device == "gpu":
            self.inputs = cp.asarray(self.inputs)

class OrthogonalSequence(Sequence):
    pass

class ExternalInput(object):
    """
    Inputs:
      - patterns:
      - Delta_t:
      - Delta_trial:
      - dt:
    """
    def __init__(self, patterns, Delta_t, Delta_trial, dt, *args, **kwargs):
        self.N = N = kwargs["N"]
        self.n_trials = n_trials = kwargs["n_trials"]
        self.P = len(patterns[0])
        self.xi = patterns
        self.Delta_t = Delta_t
        self.Delta_trial = Delta_trial
        self.xi_intertrial = np.random.RandomState(seed=199).randn(
                N, n_trials*int(Delta_trial/Delta_t))
        self.dt = dt
        
        # Convert from units of time to units of dt in order to avoid
        # floating point errors when taking modulus
        t_trial_xi = self.Delta_t*self.P
        t_trial_eta = self.Delta_trial
        t_trial = t_trial_xi + t_trial_eta
        self.n_trial_xi = np.rint(t_trial_xi/dt)
        self.n_trial_eta = np.rint(t_trial_eta/dt)
        self.n_trial = np.rint(t_trial/dt)
        self.n_Delta_t = np.rint(Delta_t/dt)
    
    def __call__(self, t):
        n = np.rint(t/self.dt)
        n_ = n%self.n_trial        
        if n_ < self.n_trial_xi:
            mu = int(n_/self.n_Delta_t) 
            return self.xi[0,mu,:] # NOTE: Valid for patterns with only one sequence (i.e. S=1)
            #return [mu]
        else:
            mu_prime = int(n/self.n_trial)*int(self.n_trial_eta/self.n_Delta_t) + \
                       int((n_-self.n_trial_xi)/self.n_Delta_t)
            return self.xi_intertrial[:,mu_prime]
            #return [mu_prime]


class OUInput(object):
    def __init__(self, mu_input, T, dt=1e-3, tau_ou=10e-3, sigma_ou=0.1):
        self.N = N = mu_input.N
        self.P = mu_input.P
        self.dt = dt
        n_T = int(np.rint(T/dt))
        p_input = np.zeros((n_T,N))
        sigma_noise = sigma_ou*np.sqrt(2/tau_ou)
        for n in range(n_T-1):
            p_input[n+1,:] = (1 - dt/tau_ou)*p_input[n,:] + dt/tau_ou*mu_input(n*dt) + \
                sigma_noise*np.sqrt(dt)*np.random.randn(N)
        self.p_input = p_input
    
    def __call__(self, t):
        return self.p_input[int(np.rint(t/self.dt)),:]
