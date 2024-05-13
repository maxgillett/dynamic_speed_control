import pdb
import time
import copy
import logging
import numpy as np
import scipy.sparse
import scipy.integrate
from scipy.optimize import minimize
from tqdm.auto import tqdm, trange
import progressbar
from numba import jit, njit, prange
from connectivity import Connectivity, SparseConnectivity, BilinearPlasticityRule, TriphasicPlasticityRule
from transfer_functions import ErrorFunction
from helpers import spike_to_rate

import importlib 
cupy_loader = importlib.find_loader('cupy')
if cupy_loader is not None:
    import cupy as cp
else:
    cp = np
    np.get_array_module = lambda x: np

logger = logging.getLogger(__name__)

class Population(object):
    def __init__(self, N, tau, phi=lambda x:x, name="exc"):
        self.name = name
        self.size = N
        self.state = np.array([], ndmin=2).reshape(N,0)
        self.field = np.array([], ndmin=2).reshape(N,0)
        self.tau = tau
        self.phi = phi


class SpikingNeurons(Population):
    def __init__(self, N, tau_mem, tau_syn, thresh, reset, name="exc"):
        super(SpikingNeurons, self).__init__(N, None, None)
        self.name = name
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.thresh = thresh
        self.reset = reset


class Network(object):
    def __init__(self, exc, inh=None, 
            c_EE=Connectivity(), c_IE=Connectivity(), 
            c_EI=Connectivity(), c_II=Connectivity(),
            device="cpu"):
        """
        """
        self.c_EE = c_EE
        self.c_IE = c_IE
        self.c_EI = c_EI
        self.c_II = c_II
        self.exc = exc
        self.inh = inh
        self.size = exc.size
        self.t = np.array([])
        if device == "cpu":
            self.W_EE = c_EE.W
            self.W_EI = c_EI.W
            self.W_IE = c_IE.W
            self.W_II = c_II.W
            self.W = scipy.sparse.bmat([
                [c_EE.W, c_EI.W],
                [c_IE.W, c_II.W]
            ]).tocsr()
        elif device == "gpu":
            self.W = c_EE.W
            if inh is not None:
                raise NotImplementedError
        if inh:
            self.tau = np.concatenate([
                np.array([exc.tau]*exc.size),
                np.array([inh.tau]*inh.size)
            ])
            self.size += self.inh.size
        else:
            self.tau = exc.tau
        self.xi = None
        self.r_ext = 0


class RateNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(), 
            c_EI=Connectivity(), c_II=Connectivity(),
            formulation=1,
            epsilon_r=1,
            device="cpu",
            monitors=[],
            disable_pbar=False):
        super(RateNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II, device)
        self.formulation = formulation
        self.monitors = monitors
        self.device = device
        self.epsilon_r = epsilon_r
        self.disable_pbar = disable_pbar
        if self.formulation == 1:
            self._fun = self._fun1
        elif self.formulation == 2:
            self._fun = self._fun2
        elif self.formulation == 3:
            self._fun = self._fun3
        elif self.formulation == 4:
            self._fun = self._fun4
        elif self.formulation == 5:
            self._fun = self._fun5

    def simulate(self, t, r0, t0=0, dt=1e-3, r_ext=lambda t: 0, save_field=True):
        """
        Runge-Kutta 2nd order
        """
        logger.info("Integrating network dynamics")
        if self.disable_pbar:
            pbar = progressbar.NullBar()
        else:
            pbar = progressbar.ProgressBar(
                maxval=t,
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        pbar.start()
        self.r_ext = r_ext
        sol = scipy.integrate.solve_ivp(
            self._fun(pbar,t),
            t_span=(t0,t),
            t_eval=np.arange(0,t,dt),
            y0=r0,
            method="RK23")
        pbar.finish()
        self.t = sol.t
        state = sol.y

        # Save network state
        self.exc.state = np.hstack([self.exc.state, state[:self.exc.size,:]])
        if self.inh: 
            self.inh.state = np.hstack([self.inh.state, state[-self.inh.size:,:]])

        if save_field:
            if self.formulation == 1:
                raise NotImplementedError
            elif self.formulation == 2:
                self.exc.field = self.W_EE.dot(self.exc.phi(self.exc.state))
                if self.inh:
                    self.exc.field += self.W_EI.dot(self.inh.phi(self.inh.state))
            else:
                raise StandardError("Unknown rate formulation")

    def simulate_euler(self, t, r0, t0=0, dt=1e-3, r_ext=lambda t: 0, save_field=True):
        """
        Euler-Maryama scheme
        """
        logger.info("Integrating network dynamics")
        if self.device == "cpu":
            self.r_ext = r_ext
            state = np.zeros((self.exc.size, int((t-t0)/dt)))
            field = np.zeros_like(state)
            state[:,0] = r0
            if self.disable_pbar:
                pbar = progressbar.NullBar()
            else:
                pbar = progressbar.ProgressBar(
                    maxval=t,
                    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            pbar.start()
            fun = self._fun(pbar,t)
            for i, t in enumerate(np.arange(t0,t-dt,dt)[:-1]):
                r = state[:,i]
                if save_field:
                    dr, field_i = fun(t, r, return_field=save_field)
                else:
                    dr = fun(t, r, return_field=save_field)
                if self.xi:
                    sigma = np.sqrt(field.var()) / self.tau
                    dr += self.xi.value(dt,self.tau,self.exc.size) * sigma
                state[:,i+1] = state[:,i] + dt * dr
                if save_field:
                    field[:,i] = field_i
            self.exc.state = np.hstack([self.exc.state, state[:self.exc.size,:]])
            if save_field:
                self.exc.field = np.hstack([self.exc.field, field[:self.exc.size,:]])
            if self.inh: 
                self.inh.state = np.hstack([self.inh.state, state[-self.inh.size:,:]])
                if save_field:
                    self.inh.field = np.hstack([self.inh.field, field[:self.inh.size,:]])
        elif self.device == "gpu":
            self.r_ext = r_ext
            # TODO: r_ext is not implemented
            if save_field:
                return NotImplementedError
            if self.xi:
                return NotImplementedError
            state = cp.zeros((self.exc.size, int((t-t0)/dt)))
            state[:,0] = r0
            if self.disable_pbar:
                pbar = progressbar.NullBar()
            else:
                pbar = progressbar.ProgressBar(
                    maxval=t,
                    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            pbar.start()
            fun = self._fun(pbar,t)
            r = cp.asarray(r0)
            for i, t in enumerate(np.arange(t0,t-dt,dt)[:-1]):
                dr = fun(t, r)
                r = r + dt * dr
                state[:,i+1] = r#r.get()
            self.exc.state = state.get()


    def add_noise(self, xi, pop):
        self.xi = xi

    def _fun1(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 1
            """
            # $ \frac{dx}{dt} = -x + \phi( \sum_{j} J_{ij} x_j + I_0 ) $

            pbar.update(t%t_max)

            if self.inh:
                raise NotImplementedError
            
            if self.device == "cpu":
                phi_r = self.exc.phi
                r_ext = self.r_ext
                r_sum = self.epsilon_r*self.W.dot(r) + r_ext(t)
                dr = (-r + phi_r(r_sum)) / self.tau
                if return_field:
                    return dr, r_sum
                else:
                    return dr
            elif self.device == "gpu":
                #pdb.set_trace()
                phi_r = self.exc.phi
                r_ext = self.r_ext
                #r_sum = self.epsilon_r*self.W.dot(r) #+ r_ext(t)
                r_sum = self.epsilon_r*self.W.dot(r) + r_ext(t)
                dr = (-r + phi_r(r_sum)) / self.tau
                return dr

        return f

    def _fun2(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 2 (inhibition optional)
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij} /phi(x_j) + I_0 $

            pbar.update(t)

            if self.inh:
                phi_r = np.zeros_like(r)
                phi_r[:self.exc.size] = self.exc.phi(r[:self.exc.size])
                phi_r[-self.inh.size:] = self.inh.phi(r[-self.inh.size:])
            else:
                phi_r = self.exc.phi(r)
            r_ext = self.r_ext
            r_sum = self.W.dot(phi_r)
            dr = (-r + r_sum + r_ext(t)) / self.tau
            if return_field:
                return dr, r_sum
            else:
                return dr
        return f

    def _fun3(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 3: instantaneous inhibition
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij} /phi(x_j) + I_inh + I_0 $

            pbar.update(t%t_max)

            if self.inh is None:
                return NotImplementedError

            #FIXME: 20 is g_phi^I. Do not hardcode
            r_ext = self.r_ext
            phi_r_exc = self.exc.phi(r[:self.exc.size])
            r_sum_exc = self.W_EE.dot(phi_r_exc)
            r_sum_inh = 20*self.W_EI.dot(self.W_IE.dot(phi_r_exc))
            dr = np.zeros(self.size)
            dr[:self.exc.size] = (-r[self.exc.size] + r_sum_exc + r_sum_inh + r_ext(t)) / \
                                 self.tau[:self.exc.size]

            return dr
        return f

    def _fun4(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 4: Effective inhibition (Eq. 46 of Gillett et al)
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij}^EE /phi(x_j) + \sum_{j} J_{ij}^I /phi(x_j) $

            pbar.update(t%t_max)

            for monitor in self.monitors:
                monitor.run(t, self.exc.phi(r))
                #monitor.run(t, r)

            r_ext = self.r_ext
            phi_r_exc = self.exc.phi(r)
            r_sum_exc = self.W_EE.dot(phi_r_exc)
            r_sum_inh = self.W_EI.dot(self.W_IE.dot(phi_r_exc))
            dr = (-r + r_sum_exc + r_sum_inh + r_ext(t)) / self.tau
            return dr

        return f

    def _fun5(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 5: Effective (global) inhibition (Eq. 46 of Gillett et al)
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij}^EE /phi(x_j) + J^I /phi(x_j) $

            pbar.update(t%t_max)

            for monitor in self.monitors:
                monitor.run(t, self.exc.phi(r))
                #monitor.run(t, r)

            r_ext = self.r_ext
            phi_r_exc = self.exc.phi(r)
            r_sum_exc = self.W_EE.dot(phi_r_exc)
            r_sum_inh = self.c_EE.J_I*np.mean(phi_r_exc)
            dr = (-r + r_sum_exc + r_sum_inh + r_ext(t)) / self.tau
            return dr

        return f

    def overlap_with(self, vec, pop, spikes=False):
        """
        Compute the overlap of network activity with a given input vector
        """
        if self.formulation == 1:
            rate = pop.state
        elif self.formulation == 2:
            rate = pop.phi(pop.state)
        elif self.formulation == 4:
            rate = pop.phi(pop.state)
        elif self.formulation == 5:
            rate = pop.phi(pop.state)
        else:
            raise StandardError("Unknown rate formulation")
        if self.device == "cpu":
            return rate.T.dot(vec) / self.exc.size
        elif self.device == "gpu":
            return rate.T.dot(vec.get()) / self.exc.size

    def clear_state(self):
        self.exc.state = np.array([], ndmin=2).reshape(self.exc.size,0)
        self.exc.field = np.array([], ndmin=2).reshape(self.exc.size,0)
        if self.inh:
            self.inh.state = np.array([], ndmin=2).reshape(self.inh.size,0)
            self.inh.field = np.array([], ndmin=2).reshape(self.inh.size,0)


class PoissonNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(), 
            c_EI=Connectivity(), c_II=Connectivity(),
            r_max=100.):
        super(PoissonNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II)
        self.r_max=r_max

    def simulate(self, t, r0, t0=0, dt=1e-3, exact=False):
        tau = self.tau
        neighbors = self.c_EE.ij
        W = self.W
        r = r0*1
        phi = self.exc.phi

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size))
        spikes = np.zeros_like(state)

        #@njit -- no speedup with compilation
        def func(state, spikes, W, r, r_max, t_size, tau, dt, phi, neighbors):
            for n in range(t_size-1):
                rv = np.random.random(size=r.size)
                spks = np.nonzero(rv < phi(r)*dt*r_max)[0]

                # Propagate spike to neighbors
                for j in spks:
                    idxs = neighbors[j]
                    r[idxs] += 1./r_max*W[idxs,j].flatten()/tau

                # Decay intensity function for all units
                r = r * np.exp(-dt/tau)

                # Record state and spikes
                state[:,n+1] = r
                spikes[spks,n] = 1

        # Much faster with dense matrix
        func(state, spikes, np.asarray(W.todense()), r, self.r_max,
             t_size, tau, dt, phi, neighbors)

        # Save network state
        self.exc.state = state[:self.exc.size,:]
        self.exc.spikes = spikes[:self.exc.size,:]
        if self.inh: 
            self.inh.state = state[self.exc.size:,:]
            self.inh.spikes = spikes[self.exc.size:,:]

    def rates(self, pop):
	    r = spike_to_rate(pop.spikes)
	    return r

    def overlap_with(self, vec, pop, spikes=False):
        "Compute the overlap of network activity with a given input vector"
        if spikes:
            r = self.rates(pop)
        else:
            r = pop.state
        return r.T.dot(vec) / self.exc.size


class SpikingNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(),
            c_EI=Connectivity(), c_II=Connectivity(),
            r_max=100.):
        super(SpikingNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II)

    def simulate(self, t, s0, v0, t0=0, dt=1e-3, tau_rp=1e-3, exact=False):
        neighbors = self.c_EE.ij
        W_EE = np.asarray(self.W_EE.todense())
        s = s0
        v = v0
        thresh = self.exc.thresh
        reset = self.exc.reset
        tau_mem = self.exc.tau_mem
        tau_syn = self.exc.tau_syn

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size, 2))
        spikes = np.zeros_like(state[:,:,0])

        for n in range(t_size-1):
            spks = np.nonzero(v > thresh)[0]

            # Decay intensity function for all units
            v = v * np.exp(-dt/tau_mem) + s * (1 - np.exp(-dt/tau_mem))
            s = s * np.exp(-dt/tau_syn)
            v[spks] = reset

            # Propagate spike to neighbors
            for j in spks:
                idxs = neighbors[j]
                s[idxs] += W_EE[idxs,j].flatten()/tau_syn

            if n > 0:
                idxs = spikes[:,n-1].nonzero()[0]
                v[idxs] = reset

            # Record state and spikes
            state[:,n+1,0] = s
            state[:,n+1,1] = v
            spikes[spks,n] = 1

        # Save network state
        self.exc.state = state[:self.exc.size,:,:]
        self.exc.spikes = spikes[:self.exc.size,:]

    # TODO: Note that refractory period is hardcoded to 1ms right now
    def simulate_two_pop(self, 
            t,
            s0_exc,
            v0_exc,
            s0_inh,
            v0_inh,
            t0=0,
            dt=1e-3,
            tau_rp=1e-3,
            sigma_lif_E=0,
            sigma_lif_I=0,
            mu_E_1=None,
            mu_E_2=None,
            v_exc_lower_bound=None,
            white_noise_seed=100):
        """
        Simulate two population LIF spiking network

        Inputs:
            t: 
            s0_exc:
            v0_exc:
            s0_exc:
            v0_exc: 
            t0:
            dt:
            tau_rp:
            v_exc_lower_bound:
        """

        neighbors_EE = self.c_EE.ij
        neighbors_EI = self.c_EI.ij
        neighbors_IE = self.c_IE.ij
        neighbors_II = self.c_II.ij

        W_EE = np.asarray(self.W_EE.todense())
        W_EI = np.asarray(self.W_EI.todense())
        W_IE = np.asarray(self.W_IE.todense())
        W_II = np.asarray(self.W_II.todense())

        s_exc = s0_exc
        v_exc = v0_exc
        s_inh = s0_inh
        v_inh = v0_inh

        thresh_exc = self.exc.thresh
        reset_exc = self.exc.reset
        tau_mem_exc = self.exc.tau_mem
        tau_syn_exc = self.exc.tau_syn

        thresh_inh = self.inh.thresh
        reset_inh = self.inh.reset
        tau_mem_inh = self.inh.tau_mem
        tau_syn_inh = self.inh.tau_syn

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size, 2))
        spikes = np.zeros_like(state[:,:,0])

        xi = np.random.RandomState(seed=white_noise_seed)

        for n in trange(t_size-1):

            # Get indices of neurons that crossed spiking threshold in the last timestep
            spks_exc = np.nonzero(v_exc >= thresh_exc)[0]
            spks_inh = np.nonzero(v_inh >= thresh_inh)[0]

            # Evolve membrane voltage in time
            v_exc = v_exc * np.exp(-dt/tau_mem_exc) + \
                    s_exc * (1 - np.exp(-dt/tau_mem_exc))
            v_inh = v_inh * np.exp(-dt/tau_mem_inh) + \
                    s_inh * (1 - np.exp(-dt/tau_mem_inh))

            # Evolve synaptic currents in time
            s_inh = s_inh * np.exp(-dt/tau_syn_inh)
            s_exc = s_exc * np.exp(-dt/tau_syn_exc)

            # Propagate exc spikes to neighbors
            for j in spks_exc:
                idxs_EE = neighbors_EE[j]
                idxs_IE = neighbors_IE[j]
                s_exc[idxs_EE] += W_EE[idxs_EE,j].flatten()/tau_syn_exc
                s_inh[idxs_IE] += W_IE[idxs_IE,j].flatten()/tau_syn_inh
            # Propagate inh spikes to neighbors
            for j in spks_inh:
                idxs_II = neighbors_II[j]
                idxs_EI = neighbors_EI[j]
                s_inh[idxs_II] += W_II[idxs_II,j].flatten()/tau_syn_inh
                s_exc[idxs_EI] += W_EI[idxs_EI,j].flatten()/tau_syn_exc

            # Reset neurons that spiked in last timestep
            v_exc[spks_exc] = reset_exc
            v_inh[spks_inh] = reset_inh

            # Inject white noise into membrane potentials
            if sigma_lif_E > 0:
                v_exc += np.sqrt(dt/tau_mem_exc)*sigma_lif_E*xi.normal(size=self.exc.size)
            if sigma_lif_I > 0:
                v_inh += np.sqrt(dt/tau_mem_inh)*sigma_lif_I*xi.normal(size=self.inh.size)

            # Inject DC input
            N = v_exc.size
            if mu_E_1:
                v_exc[:int(N/2)] += mu_E_1
            if mu_E_2:
                v_exc[int(N/2):] += mu_E_2
                
            # Clamp membrane potentials above threshold to threshold
            v_exc[np.nonzero(v_exc > thresh_exc)[0]] = thresh_exc
            v_inh[np.nonzero(v_inh > thresh_inh)[0]] = thresh_inh

            # Clamp membrane potentials below voltage floor to floor
            if v_exc_lower_bound:
                v_exc[v_exc < v_exc_lower_bound] = v_exc_lower_bound

            # Refractory period: Clamp voltage of neurons that spiked 1 timestep ago to reset/rest
            if n > 0:
                idxs_exc = spikes[:,n-1][:self.exc.size].nonzero()[0]
                idxs_inh = spikes[:,n-1][self.exc.size:].nonzero()[0]
                v_exc[idxs_exc] = reset_exc
                v_inh[idxs_inh] = reset_inh

            # Record state and spikes
            state[:,n+1,0] = np.concatenate([s_exc,s_inh])
            state[:,n+1,1] = np.concatenate([v_exc,v_inh])
            spikes[spks_exc,n] = 1
            spikes[self.exc.size+spks_inh,n] = 1

        # Save network state
        self.exc.state = state[:self.exc.size,:,:]
        self.exc.spikes = spikes[:self.exc.size,:]
        if self.inh: 
            self.inh.state = state[self.exc.size:,:,:]
            self.inh.spikes = spikes[self.exc.size:,:]

    def rates(self, pop):
	    r = spike_to_rate(pop.spikes)
	    return r

    def overlap_with(self, vec, pop, spikes=False):
        "Compute the overlap of network activity with a given input vector"
        if spikes:
            r = self.rates(pop)
        else:
            r = pop.state[:,:,0]
        return r.T.dot(vec) / self.exc.size



class LearningProcess(object):
    def __init__(self, *args, **kwargs):
        pass


class DiscreteLearningProcess(LearningProcess):
    def __init__(self, phi, sequences, patterns, params):
        self.phi = phi
        self.sequences = sequences
        self.patterns = patterns
        self.measurements = []
        
        self.tau          = params.get('tau',          10e-3)
        self.N            = params.get('N',            5000)
        self.K            = params.get('K',            200)
        self.A            = params.get('A',            0.5)
        self.measure_t    = params.get('measure_t',    0.4) # Measurement time (s)
        self.sigma_J      = params.get('sigma_J',      0.05)
        self.epsilon_r    = params.get('epsilon_r',    0)
        self.Delta_t      = params.get('Delta_t',      1)
        self.G_min        = params.get('G_min',        -np.inf)
        self.G_max        = params.get('G_max',        np.inf)
        self.dt           = params.get('dt',           1e-3)
        self.L_pre        = params.get('L_pre',        None) # Lamba vector length
        self.L_post       = params.get('L_post',       None) # Lamba vector length
        self.beta         = params.get('beta',         1) # Lamba vector length
        self.I_ext        = params.get('I_ext',        0) # Lamba vector length
        self.I_delta_max  = params.get('I_delta_max',  0.05) 
        self.sigma_z      = params.get('sigma_z',      0) 
        self.alpha_G      = params.get('alpha_G',      0.5)
        self.device       = params.get('device',       'cpu')
        self.mat_type     = params.get('mat_type',     'coo')
        self.reset_r      = params.get('reset_r',      False)
        self.coding_level = params.get('coding_level', 0.1)
        self.record_v_mat = params.get('record_v_mat', False)

        self.update_rule_version          = params.get('update_rule_version',          'v1')
        self.random_connectivity          = params.get('random_connectivity',          True)
        self.hebbian_active               = params.get('hebbian_active',               True)
        self.hebbian_bilinear             = params.get('hebbian_bilinear',             False)
        self.rescaling_active             = params.get('rescaling_active',             True)
        self.rescaling_field_active       = params.get('rescaling_field_active',       True)
        self.metaplasticity_active        = params.get('metaplasticity_active',        False)
        self.metaplastic_threshold_active = params.get('metaplastic_threshold_active', False)

        #if self.hebbian_bilinear:
        #    raise NotImplementedError
        
        # Initialize population and connectivity
        c = self.K/self.N
        self.pop = pop = Population(N=self.N, tau=self.tau, phi=phi)
        self.conn = conn = SparseConnectivity(pop, pop, p=c,
            sparse_mat_type=self.mat_type,
            fixed_degree=True,
            device=self.device,
            disable_pbar=True)
        conn.set_all(1)
        conn.W.data[:] = 0
        if self.random_connectivity:
            if self.device == "cpu":
                conn.W.data += self.sigma_J*np.random.randn(conn.W.data.size)
            elif self.device == "gpu":
                conn.W.data += self.sigma_J*cp.random.randn(conn.W.data.size)
        
    def attach_measurement(self, measurement):
        measurement.attach(self)
        self.measurements.append(measurement)

    def dq_dlambda(self, conn, rule, G, epsilon_r=1, L_pre=None, L_post=None):
        if self.device == "cpu":
            xp = np
        elif self.device == "gpu":
            xp = cp
        N = self.N
        dt = self.dt
        tau = self.tau
        phi = self.phi
        Delta_t = self.Delta_t
        patterns = self.patterns
        P = patterns.shape[1]
        if L_pre is None:
            L_pre = P
        if L_post is None:
            L_post = P
        dq = np.zeros((P,P,L_pre+L_post))
        for l in range(L_pre+L_post):
            lambda_pre = xp.zeros(int(L_pre))
            lambda_post = xp.zeros(int(L_post))
            if l < L_pre:
                lambda_pre[l] = 1e-3
            else:
                lambda_post[l-L_pre] = 1e-3
            r_mat = xp.zeros((P*Delta_t, N))
            conn_W_copy = conn.W.copy()
            J_data = conn_W_copy.data #conn.W.data.copy()
            dJ_data = xp.zeros_like(J_data)
            dJ_data_sum = xp.zeros_like(J_data)
            for mu_ in range(P):
                for t in range(Delta_t):
                    #dJ_data[:] = 0
                    r_mat = xp.roll(r_mat, shift=1, axis=0)
                    if self.device == "cpu":
                        if Delta_t == 1:
                            r_mat[0,:] = phi(
                                np.sqrt(1-epsilon_r**2)*patterns[0][mu_] + 
                                epsilon_r*conn.dot(J_data, phi(patterns[0][mu_])))
                        else:
                            if mu_ == 0 and t == 0:
                                h_t_rec = conn.dot(J_data, phi(self.patterns[0][mu_]))
                            else:
                                h_t_rec = conn.dot(J_data, r_mat[1,:])
                            v = np.sqrt(1-epsilon_r**2)*patterns[0][mu_] + \
                                         epsilon_r*h_t_rec
                            r_mat[0,:] = r_mat[1,:]*(1-dt/tau) + \
                                         dt/tau*phi(v)
                    elif self.device == "gpu":
                        if Delta_t == 1:
                            r_mat[0,:] = phi(
                                np.sqrt(1-epsilon_r**2)*patterns[0][mu_] + 
                                epsilon_r*conn_W_copy.dot(phi(patterns[0][mu_])))
                        else:
                            if mu_ == 0 and t == 0:
                                h_t_rec = conn_W_copy.dot(phi(self.patterns[0][mu_]))
                            else:
                                h_t_rec = conn_W_copy.dot(r_mat[1,:])
                            v = np.sqrt(1-epsilon_r**2)*patterns[0][mu_] + \
                                         epsilon_r*h_t_rec
                            r_mat[0,:] = r_mat[1,:]*(1-dt/tau) + \
                                         dt/tau*phi(v)

                    if self.hebbian_active > 0:
                        if (t+1) % self.hebbian_active == 0:
                            if mu_ > 0:
                                rule.update(
                                    dJ_data_sum,
                                    dJ_data,
                                    lambda_pre,
                                    lambda_post,
                                    r_mat,
                                    Delta_t,
                                    dt)
            if self.device == "cpu":
                J_data += dJ_data_sum
            else:
                conn_W_copy.data += dJ_data_sum

            for mu in range(P):
                phi_xi = phi(patterns[0][mu])
                if self.device == "cpu":
                    J0_phi_xi = conn.dot(conn.W.data, phi_xi)
                    J1_phi_xi = conn.dot(J_data, phi_xi)
                elif self.device == "gpu":
                    J0_phi_xi = conn.W.dot(phi_xi)
                    J1_phi_xi = conn_W_copy.dot(phi_xi)
                for mu_prime in range(P):
                    q0 = patterns[0][mu_prime].dot(J0_phi_xi)/N
                    q1 = patterns[0][mu_prime].dot(J1_phi_xi)/N
                    dq[mu, mu_prime, l] = (float(q1)-float(q0))/1e-3
        return dq

    def optimal_lambda_vec(self, dq, L_pre=None, L_post=None, beta=1, debug=False):
        P = self.patterns.shape[1]
        dq_optimal = np.diag(np.ones(P), k=1)[:P,:P]
        # If the final pattern is zero, then remove appropriate 
        # entry from target dq matrix
        if self.patterns[0,-1,:].nonzero()[0].size == 0:
            dq_optimal[-2,-1] = 0
        dq_optimal *= 1
        def fun(w_vec):
            dq_sum = np.zeros((P,P))
            dq_sum2 = np.zeros(P-1)
            for n, w in enumerate(w_vec):
                dq_sum += w*dq[:,:,n]
                dq_sum2 += np.diagonal(w*dq[:,:,n], offset=1)
            f1 = np.sum((dq_optimal-dq_sum)**2)
            f2 = -np.sum(dq_sum2)
            return beta*f1 + (1-beta)*f2
        if L_pre is None:
            L_pre = P
        if L_post is None:
            L_post = P
        w0_vec = np.zeros(L_pre+L_post) 
        w0_vec[1] = 1
        res = minimize(fun, w0_vec)
        if debug:
            w_vec = res['x']
            dq_sum = np.zeros((P,P))
            dq_sum2 = np.zeros(P-1)
            for n, w in enumerate(w_vec):
                dq_sum += w*dq[:,:,n]
                dq_sum2 += np.diagonal(w*dq[:,:,n], offset=1)
            f1 = np.sum((dq_optimal-dq_sum)**2)
            f2 = -np.sum(dq_sum2)
            print(beta*f1, (1-beta)*f2)
        #if debug:
        #    del res['hess_inv']
        #    del res['jac']
        #    print(res)
        return res['x']/res['x'].max(), res

    def run(self, n_trials,
            lambda_pre, lambda_post,
            hebbian_rule,
            rescaling_rule,
            transfer_function,
            disable_pbar=False, debug=False):

        pop = self.pop
        conn = self.conn
        tau = self.tau
        sequences = self.sequences
        patterns = self.patterns
        P = self.patterns.shape[1]
        Delta_t = self.Delta_t
        N = pop.size
        L_pre = self.L_pre
        L_post = self.L_post
        beta = self.beta
        I_ext = self.I_ext
        I_delta_max = self.I_delta_max
        dt = self.dt
        sigma_J = self.sigma_J
        epsilon_r = self.epsilon_r
        miniters = int(n_trials/10)

        # Quenched external input added during recall
        sigma_z = self.sigma_z
        z = sigma_z*np.random.randn(N)
        
        # Truncate lambda vectors if metaplasticity is not enabled
        if not self.metaplasticity_active:
            lambda_pre_size = lambda_pre.nonzero()[0][-1]+1
            lambda_pre = lambda_pre[:lambda_pre_size] 
            if L_post > 0:
                lambda_post_size = lambda_post.nonzero()[0][-1]+1
                lambda_post = lambda_post[:lambda_post_size] 
            else:
                lambda_post = np.zeros(0)

        # Construct buffers for weight updates
        h_mat, r_mat = self.construct_buffers(P*Delta_t, pop.size)
        xp = cp.get_array_module(r_mat)
        dJ_data = xp.zeros_like(conn.W.data)
        G = xp.ones(int(pop.size))
        G_prev = xp.ones(int(pop.size))
        if self.hebbian_bilinear or self.record_v_mat:
            v_mat = xp.zeros((P*Delta_t, pop.size))

        # Manipulate in place the data of the sparse connectivity matrix representation 
        for n in trange(0, n_trials, miniters=miniters, disable=disable_pbar):
            self.reset_buffers(h_mat, r_mat)
            for mu in range(P):
                for t in range(Delta_t):
                    h_mat, r_mat = self.roll_buffers(h_mat, r_mat)
                    if Delta_t == 1:
                        h_t_rec = conn.dot(conn.W.data, pop.phi(self.patterns[0][mu]))
                        h_mat[0,:] = h_t_rec
                        v          = np.sqrt(1-epsilon_r**2)*patterns[0][mu] + \
                                     epsilon_r*h_t_rec
                        r_mat[0,:] = pop.phi(v)
                    else:
                        if mu == 0 and t == 0:
                            h_t_rec = conn.dot(conn.W.data, pop.phi(self.patterns[0][mu]))
                        else:
                            h_t_rec = conn.dot(conn.W.data, r_mat[1,:])
                        h_mat[0,:] = h_t_rec
                        v          = np.sqrt(1-epsilon_r**2)*patterns[0][mu] + \
                                     epsilon_r*h_t_rec + \
                                     I_ext
                        r_mat[0,:] = r_mat[1,:]*(1-dt/tau) + dt/tau*pop.phi(v)

                        if self.reset_r:
                            if t == 0:
                                r_mat[0,:] = 0

                    if self.hebbian_bilinear or self.record_v_mat:
                        v_mat = np.roll(v_mat, shift=1, axis=0)
                        v_mat[0,:] = v

                    if self.hebbian_active > 0:
                        if (t+1) % self.hebbian_active == 0:
                            if mu > 0:
                                if self.hebbian_bilinear:
                                    hebbian_rule.update(
                                            conn.W.data,
                                            dJ_data,
                                            lambda_pre,
                                            lambda_post,
                                            v_mat,
                                            Delta_t,
                                            dt)
                                else:
                                    if self.update_rule_version == "v1":
                                        hebbian_rule.update(
                                                conn.W.data,
                                                dJ_data,
                                                lambda_pre,
                                                lambda_post,
                                                r_mat,
                                                Delta_t,
                                                dt)
                                    elif self.update_rule_version == "v2":
                                        hebbian_rule.update_v2(
                                                conn.W.data,
                                                dJ_data,
                                                lambda_pre,
                                                lambda_post,
                                                r_mat,
                                                v_mat,
                                                Delta_t,
                                                dt)


            # Metaplasticity of neuron specific external input
            if self.metaplastic_threshold_active:
                if n == 0:
                    self.v_mat0 = v_mat.copy()
                    I_ext = xp.zeros(int(N))
                    self.I_ext = I_ext
                else:
                    v_mat_mean = xp.mean(-0.0-v_mat, axis=0)
                    I_ext_delta = xp.clip(v_mat_mean, a_min=-I_delta_max, a_max=I_delta_max)
                    I_ext += I_ext_delta
                    I_ext = xp.clip(I_ext, a_min=-0.4, a_max=0.6)
                    self.I_ext = I_ext

            # Constant incoming weight variance homeostatic rule
            if self.rescaling_active:
                rescaling_rule.update(conn.W.data, sigma_J)

            # Constant field variance (over pattern presentations) homeostatic rule
            if self.rescaling_field_active > 0:
                if n % self.rescaling_field_active == 0:
                    if self.device == "cpu":
                        var = np.sqrt(h_mat.var(axis=0))
                    elif self.device == "gpu":
                        var = cp.sqrt(h_mat.var(axis=0))
                    if n == 0:
                        G[:] = 1/var
                    else:
                        G[:] = self.alpha_G/var + (1-self.alpha_G)*G_prev # Exponential MA
                        #G[:] = 1/np.sqrt((h_mat).var(axis=0))            # No history dependence
                    #G[:] = np.clip(G, a_min=self.G_min, a_max=self.G_max)
                    G_prev[:] = G.copy()
                    for i in range(N):
                        conn.W.data[conn.a[i]:conn.a[i+1]] *= G[i]

            # Optimize weight kernel to grow shifted diagonal structure in $Q_mu_mu^\prime$
            if self.metaplasticity_active:
                dq = self.dq_dlambda(conn, hebbian_rule, G, epsilon_r, L_pre, L_post)
                lambda_optimal, _ = self.optimal_lambda_vec(
                        dq,
                        L_pre=L_pre,
                        L_post=L_post,
                        beta=beta,
                        debug=debug)
                lambda_pre[:] = 0
                lambda_post[:] = 0
                for i, lambda_ in enumerate(lambda_optimal):
                    if i < L_pre:
                        if abs(lambda_) > 0.005:
                            lambda_pre[i] = lambda_
                    else:
                        if abs(lambda_) > 0.005:
                            lambda_post[i-L_pre] = lambda_

            # Measure quantities of interest
            measurements = [mes for mes in self.measurements if n % mes.n == 0]
            simulation_required = np.any([mes.simulation_required for mes in measurements])
            for epsilon_c in np.unique([mes.epsilon_c for mes in measurements]):
                if simulation_required:
                    if epsilon_c is None:
                        epsilon_c = epsilon_r
                    net = RateNetwork(
                            pop,
                            c_EE=conn,
                            formulation=1,
                            device=self.device,
                            epsilon_r=epsilon_c,
                            disable_pbar=True)
                    net.clear_state()
                    net.simulate_euler(
                            t=self.measure_t, #0.4,
                            r0=pop.phi(patterns[0,0,:]),
                            #r0=pop.phi(hebbian_rule.f(patterns[0,0,:])),
                            r_ext=lambda t: I_ext,#z,
                            save_field=False)
                    overlaps = sequences[0].overlaps(
                        net, pop, phi=pop.phi, plasticity=hebbian_rule, correlation=False,
                        disable_pbar=True)
                    correlations = sequences[0].overlaps(
                        net, pop, phi=pop.phi, plasticity=hebbian_rule, correlation=True,
                        disable_pbar=True)

                for measurement in [mes for mes in measurements if mes.epsilon_c == epsilon_c]:
                    if measurement.simulation_required:
                        measurement.run(n, overlaps, correlations)
                    else:
                        measurement.run(n, 
                                G=G, 
                                lambda_pre=lambda_pre,
                                lambda_post=lambda_post,
                                h_mat=h_mat,
                                r_mat=r_mat)

    def construct_buffers(self, t_size, n_size):
        if self.device == "cpu":
            h_mat = np.zeros((t_size, n_size))
            r_mat = np.zeros_like(h_mat)
        elif self.device == "gpu":
            h_mat = cp.zeros((t_size, n_size))
            r_mat = cp.zeros_like(h_mat)
        else:
            raise NotImplementedError
        return h_mat, r_mat
    
    def reset_buffers(self, h_mat, r_mat):
        h_mat.fill(0)
        r_mat.fill(0)

    def roll_buffers(self, h_mat, r_mat):
        h_mat = np.roll(h_mat, shift=1, axis=0)
        r_mat = np.roll(r_mat, shift=1, axis=0)
        return h_mat, r_mat

