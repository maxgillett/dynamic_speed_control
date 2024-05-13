# GPU implementation using CuPy

import os, itertools
import cupy as cp
import numpy as np
from scipy.stats import norm
from numba import jit, njit, prange
from tqdm import trange, tqdm

class BilinearPlasticityRule:
    def __init__(self):
        self.f = lambda x:x
        self.g = lambda x:x
        
class ThresholdPlasticityRule:
    def __init__(self,
            x_f,
            q_f,
            x_g=None,
            q_g=None,
            rv=norm):
        if not x_g: x_g = x_f
        if not q_g: q_g = rv.cdf(x_g)
        self.x_f, self.q_f = x_f, q_f
        self.x_g, self.q_g = x_g, q_g
        def f(x):
            return np.where(x > x_f, q_f, -(1-q_f))
        def g(x):
            return np.where(x > x_g, q_g, -(1-q_g))
        self.f = f
        self.g = g
        
def build_ji(k, N):
    ji = []
    arr = np.arange(N)
    for i in trange(N):
        tmp = np.concatenate((arr[:i], arr[i+1:]))
        j_subset = np.random.choice(N, size=k[i], replace=False)
        ji.append(np.asarray(sorted(j_subset)))
    return ji

def weight(seq, p, f, g, i, j):
    "$f(xi_i^{\mu+p}) * g(xi_j^{\mu})$"
    if p == 0:
        return np.sum(f(seq[:,i][:,np.newaxis]) * g(seq[:,j]), axis=0)
    else:
        return np.sum(f(seq[p:,i][:,np.newaxis]) * g(seq[:-p,j]), axis=0)

def store_associations(seq, f, g, ji, K, p=1, mask=None):
    "outputs in csr representation"
    N = len(ji)
    indptr = np.asarray([0]+[len(ji[i]) for i in range(N)]).cumsum()
    indices = np.concatenate(ji)
    data = []
    for i in trange(N):
        j = ji[i]
        w = weight(seq, p, f, g, i, j) / K
        if mask is not None:
            w *= mask[i]
        data.extend(w)
    return indptr, indices, data

def reweight(indptr, indices, data, A, N, w_11, w_12, w_21, w_22):
    N1 = int(N/2)
    for n, (idx_0, idx_1) in tqdm(enumerate(zip(indptr[:N], indptr[1:N+1]))):
        idxs = indices[idx_0:idx_1]
        idx_m = idx_0 + idxs[idxs < N1].size
        if n < N1:
            data[idx_0:idx_m] *= A*w_11
            data[idx_m:idx_1] *= A*w_12
        else:
            data[idx_0:idx_m] *= A*w_21
            data[idx_m:idx_1] *= A*w_22

def erf(x):
    z = abs(x)
    t = 1.0/(1.0+0.5*z)
    ans = t*cp.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+ \
        t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+ \
        t*(-0.82215223+t*0.17087277)))))))))
    return 1 - cp.where(x > 0, ans, 2.0-ans)

def phi(x, theta, sigma):
    return 0.5 * (1 + erf((x - theta) / (cp.sqrt(2) * sigma)))

def simulate_euler(t0, T, dt, tau, N, r0, W, theta, sigma, I_ext=lambda x: 0, disable_pbar=False):
    "$\frac{dx}{dt} = -x + \phi( \sum_{j} J_{ij} x_j + I_0 )$"
    state = cp.zeros((N, int((T-t0)/dt)))
    state[:,0] = r0
    for i, t in enumerate(tqdm(np.arange(t0,T-dt,dt), disable=disable_pbar)):
        r = state[:,i]
        r_sum = W.dot(r) + I_ext(t)
        dr = (-r + phi(r_sum, theta, sigma)) / tau
        state[:,i+1] = r + dt * dr
    return state

def simulate(T, dt, tau, N, I_ext_1, I_ext_2, theta, sigma, patterns, W,
             r0=None,
             I_ext=None,
             disable_pbar=False):
    
    if I_ext is None:
        def I_ext(t):
            return cp.concatenate((
                cp.full(shape=(int(N/2)), fill_value=I_ext_1, dtype=cp.float32),
                cp.full(shape=(int(N/2)), fill_value=I_ext_2, dtype=cp.float32)))
        
    if r0 is None:
        r0 = phi(cp.asarray(patterns[0,:]), theta, sigma)
    
    r = simulate_euler(
        0, T, dt,
        tau,
        N,
        r0,
        W,
        theta, sigma,
        I_ext,
        disable_pbar)
    return r.get()

xp = cp
def correlations(r, patterns, individual=False):
    "Assumes populations of equal size"
    P, N, T = patterns.shape[0], r.shape[0], r.shape[1]
    n1 = int(N/2)
    n2 = int(N/2)
    r = xp.asarray(r)
    q = xp.zeros(shape=(P,T)) 
    q1 = xp.zeros(shape=(P,T))
    q2 = xp.zeros(shape=(P,T))
    for u, pattern in enumerate(xp.asarray(patterns)):
        # q: Correlation of whole population
        pattern_mean = pattern.mean()
        pattern_std = xp.sqrt(xp.sum((pattern-pattern_mean)**2))
        for t in range(T):
            q[u,t] = xp.sum((pattern - pattern_mean) * (r[:,t] - r[:,t].mean())) / \
                xp.sqrt(xp.sum((r[:,t]-r[:,t].mean())**2)) / \
                pattern_std
        if individual:
            # q_1: Population 1 correlations
            pattern_mean = pattern[:n1].mean()
            pattern_std = xp.sqrt(xp.sum((pattern[:n1]-pattern_mean)**2))
            for t in range(T):
                q1[u,t] = xp.sum((pattern[:n1] - pattern_mean) * (r[:n1,t] - r[:n2,t].mean())) / \
                    xp.sqrt(xp.sum((r[:n1,t]-r[:n1,t].mean())**2)) / \
                    pattern_std
            # q_2: Population 2 correlations
            pattern_mean = pattern[n1:].mean()
            pattern_std = xp.sqrt(xp.sum((pattern[n1:]-pattern_mean)**2))
            for t in range(T):
                q2[u,t] = xp.sum((pattern[n1:] - pattern_mean) * (r[n1:,t] - r[n1:,t].mean())) / \
                    xp.sqrt(xp.sum((r[n1:,t]-r[n1:,t].mean())**2)) / \
                    pattern_std
    return q.get(), q1.get(), q2.get()