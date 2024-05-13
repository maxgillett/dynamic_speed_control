import sys
import copy
import numpy as np
import warnings
from scipy.special import erf
from scipy.stats import pearsonr
sys.path.insert(0, '/home/mhg19/Thesis/network')
from network import Population, RateNetwork
from transfer_functions import ErrorFunction, ReLU
from connectivity import SparseConnectivity, LinearSynapse, ThresholdPlasticityRule, RectifiedSynapse
from sequences import GaussianSequence
from measurements import SequenceScore
from monitors import PopulationRateMonitor, MaximalFiringError, MinimalFiringError

warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficent is not defined.")

class BinarySearch():
    def __init__(self, routine, params, score, debug=False):
        self.params = copy.deepcopy(params)
        self.routine = routine
        self.debug = debug
        self.score = score

    def run(self):
        pkey = self.pkey
        debug = self.debug
        simulate = self.routine
        params = self.params
        p_lower = self.range[0]
        p_upper = self.range[1]
        p_midpoint = (p_upper + p_lower)/2.
        
        # Initial lower bound value
        params.update(pkey, p_lower)
        p_lower_val, err_msg_lower = self.score(simulate(params))
        if debug:
            print("p_lower",
                p_lower_val,
                err_msg_lower,
                p_lower,
                "[%.5f,%.5f]"%(p_lower, p_upper))
            
        # Initial upper bound value
        params.update(pkey, p_upper)
        p_upper_val, err_msg_upper = self.score(simulate(params))
        if debug:
            print("p_upper",
                p_upper_val,
                err_msg_upper,
                p_upper,
                "[%.5f,%.5f]"%(p_lower, p_upper))
        
        # Midpoint value
        p_midpoint = (p_upper + p_lower)/2.
        params.update(pkey, p_midpoint)
        p_midpoint_val, err_msg_midpoint = self.score(simulate(params))
        if debug:
            print("p_midpoint",
                p_midpoint_val,
                err_msg_midpoint,
                p_midpoint,
                "[%.5f,%.5f]"%(p_lower, p_upper))

        errcode = 0
        while True:
            # Converged
            if np.abs(p_upper-p_lower) <= self.tol:
                break

            # Midpoint left
            p_midpoint_left = (p_lower + p_midpoint)/2.
            params.update(pkey, p_midpoint_left)
            p_midpoint_left_val, err_msg_midpoint_left = self.score(simulate(params))
            if debug:
                print(
                    "p_midpoint_left",
                    p_midpoint_left_val,
                    err_msg_midpoint_left,
                    p_midpoint_left,
                    "[%.5f,%.5f]"%(p_lower, p_upper))

            # Midpoint right
            p_midpoint_right = (p_upper + p_midpoint)/2.
            params.update(pkey, p_midpoint_right)
            p_midpoint_right_val, err_msg_midpoint_right = self.score(simulate(params))
            if debug:
                print("p_midpoint_right",
                    p_midpoint_right_val,
                    err_msg_midpoint_right,
                    p_midpoint_right,
                    "[%.5f,%.5f]"%(p_lower, p_upper))

            # If firing rate exceeded bounds, then respond to error message
            if err_msg_midpoint_left or err_msg_midpoint_right:
                # Firing rate is too high
                if err_msg_midpoint_right == "high":
                    p_upper = p_midpoint_right
                    p_upper_val = np.inf
                if err_msg_midpoint_left == "high":
                    p_upper = p_midpoint_left
                    p_upper_val = np.inf

                # Firing rate is too low
                if err_msg_midpoint_right == "low":
                    p_lower = p_midpoint_right
                    p_lower_val = -np.inf
                if err_msg_midpoint_left == "low":
                    p_lower = p_midpoint_left
                    p_lower_val = -np.inf
            else:
                if p_midpoint_right_val > p_midpoint_left_val:
                    p_lower = p_midpoint_left
                    p_lower_val = p_midpoint_left_val
                else:
                    p_upper = p_midpoint_right
                    p_upper_val = p_midpoint_right_val

            p_midpoint = (p_upper + p_lower)/2.

        return {
            'errcode': errcode,
            'p_lower': p_lower,
            'p_upper': p_upper,
            'p_lower_val': p_lower_val,
            'p_upper_val': p_upper_val,
        }

def run_simulation(params):
    N_E = params.neuron['N_E']
    N_I = params.neuron['N_I']
    c = params.neuron['c']
    tau = params.neuron['tau']
    T = 0.4
    S, P = 1, 32

    n_days = params.n_days
    record_on_days = np.atleast_1d(params.record_on_days)
    
    g = params.neuron['g']
    inhibition = params.inhibition
    
    A = params.plasticity['A']
    x_f = params.plasticity['x_f']
    E_x_f = params.plasticity['E_x_f']
    q_f = 0.5*erf(x_f/np.sqrt(2)) + 0.5 + E_x_f
    
    lambda_ = params.perturbation['lambda']
    sigma_z = params.perturbation['sigma_z']
    
    exc = Population(N_E, tau=1e-2, phi=ReLU(g=g).phi)
    conn = SparseConnectivity(source=exc, target=exc, p=c, disable_pbar=True)
    sequences = [GaussianSequence(P,exc.size,seed=i) for i in range(S)]
    patterns = np.stack([s.inputs for s in sequences])
    plasticity = ThresholdPlasticityRule(x_f=x_f, q_f=q_f)

    monitor1 = PopulationRateMonitor(r_max=1000, r_min=1e-5)
    
    if inhibition == 'global': # Dense
        omega_g = params.omega['g']
        omega_o = params.omega['o']
        inh = Population(N_I, tau=tau, phi=ReLU(g=g).phi)
        K_EE, K_IE, K_EI, K_II = conn.K, N_E, N_E, 0
        synapse = RectifiedSynapse(
            K_EE, K_IE, K_EI, K_II,
            alpha=(P-1)*S/float(K_EE),
            plasticity=plasticity,
            A=A,
            g=omega_g,
            o=omega_o,
            sigma_z=sigma_z)
        conn.J_I = -np.sqrt(K_EE)*synapse.E_w
    elif inhibition == 'sparse':
        omega_g = params.omega['g']
        omega_o = params.omega['o']
        inh = Population(N_I, tau=tau, phi=ReLU(g=g).phi)
        conn_IE = SparseConnectivity(
                source=exc,
                target=inh,
                p=c,
                seed=111,
                disable_pbar=True)
        conn_EI = SparseConnectivity(
                source=inh,
                target=exc,
                p=c,
                seed=112,
                disable_pbar=True)
        K_EE, K_IE, K_EI, K_II = conn.K, conn_IE.K, conn_EI.K, 0
        synapse = RectifiedSynapse(
            K_EE, K_IE, K_EI, K_II,
            alpha=(P-1)*S/float(K_EE),
            plasticity=plasticity,
            A=A,
            g=omega_g,
            o=omega_o,
            sigma_z=sigma_z)
        conn_IE.set_all(synapse.h_IE(1))
        conn_EI.set_all(synapse.h_EI(1))
    else:
        synapse = LinearSynapse(conn.K, A=A)

    conn.store_sequences(patterns, lambda x: x, plasticity.f, plasticity.g)
    
    errcode = 0
    state, overlaps, correlations = [], [], []
    W_sequence = np.copy(conn.W.data)
    W_pert = np.zeros_like(W_sequence) 
    rng = np.random.RandomState(seed=47)
    for n in range(0, n_days):
        z_n = rng.normal(scale=1, size=conn.W.data.size)
        if n == 0:
            W_pert = sigma_z*z_n
        else:
            W_pert = lambda_*W_pert + np.sqrt(1-lambda_**2)*sigma_z*z_n
            
        #print("stds (seq, rand):", W_sequence.std(), W_pert.std())
        conn.W.data = synapse.h_EE(W_sequence + W_pert)
        
        disable_pbar = True
        if n in record_on_days:
            if inhibition == 'global':
                net = RateNetwork(
                    exc,
                    c_EE=conn,
                    formulation=5,
                    monitors=[monitor1],
                    disable_pbar=disable_pbar)
            elif inhibition == 'sparse':
                net = RateNetwork(
                    exc,
                    c_EE=conn,
                    c_EI=conn_EI,
                    c_IE=conn_IE,
                    formulation=4,
                    monitors=[monitor1],
                    disable_pbar=disable_pbar)
            else:
                net = RateNetwork(
                    exc,
                    c_EE=conn,
                    formulation=2,
                    monitors=[monitor1],
                    disable_pbar=disable_pbar)
            net.clear_state()
            try: 
                net.simulate_euler(
                    T,
                    r0=exc.phi(plasticity.f(patterns[0,0,:])),
                    save_field=False)
            except MaximalFiringError as err:
                print("errcode 1")
                errcode = 1
                break
            except MinimalFiringError as err:
                print("errcode 2")
                errcode = 2
                break
            except:
                print("errcode 3")
                errcode = 3
                break
            M = net.exc.phi(net.exc.state).mean(axis=0)
            m = sequences[0].overlaps(
                net,
                exc,
                plasticity=plasticity,
                correlation=False,
                disable_pbar=disable_pbar)
            rho = sequences[0].overlaps(
                net,
                exc,
                plasticity=plasticity,
                correlation=True,
                disable_pbar=disable_pbar)
            state.append(net.exc.state.astype(np.float32))
            overlaps.append(m.astype(np.float32))
            correlations.append(rho.astype(np.float32))
            
    # If there is an error (i.e. exploding firing rate), then don't compute scores
    if errcode == 0:
        seq_score = SequenceScore._score2(correlations[0]*M)

        if len(record_on_days) == 1:
            state_rho = np.NaN
        else:
            state_rho = np.nanmean([
                pearsonr(
                    exc.phi(state[0][i]),
                    exc.phi(state[-1][i]))[0] for i in range(N_E)])
    else:
        state_rho = np.NaN
        seq_score = np.NaN
    
    return {
        'errcode': errcode,
        'scores': {
            'state': state_rho,
            'sequence': seq_score,
        },
        'state': state,
        'overlaps': overlaps,
        'correlations': correlations,
    }

def weight_dynamics(params):
    N_E = params.neuron['N_E']
    N_I = params.neuron['N_I']
    c = params.neuron['c']
    tau = params.neuron['tau']
    T = 0.4
    S, P = 1, 32

    n_days = params.n_days
    record_on_days = np.atleast_1d(params.record_on_days)
    
    g = params.neuron['g']
    inhibition = params.inhibition
    
    A = params.plasticity['A']
    x_f = params.plasticity['x_f']
    E_x_f = params.plasticity['E_x_f']
    q_f = 0.5*erf(x_f/np.sqrt(2)) + 0.5 + E_x_f
    
    lambda_ = params.perturbation['lambda']
    sigma_z = params.perturbation['sigma_z']
    
    exc = Population(N_E, tau=1e-2, phi=ReLU(g=g).phi)
    conn = SparseConnectivity(source=exc, target=exc, p=c, disable_pbar=False)
    sequences = [GaussianSequence(P,exc.size,seed=i) for i in range(S)]
    patterns = np.stack([s.inputs for s in sequences])
    plasticity = ThresholdPlasticityRule(x_f=x_f, q_f=q_f)

    monitor1 = PopulationRateMonitor(r_max=1000, r_min=1e-5)
    
    if inhibition == 'global': # Dense
        omega_g = params.omega['g']
        omega_o = params.omega['o']
        inh = Population(N_I, tau=tau, phi=ReLU(g=g).phi)
        K_EE, K_IE, K_EI, K_II = conn.K, N_E, N_E, 0
        synapse = RectifiedSynapse(
            K_EE, K_IE, K_EI, K_II,
            alpha=(P-1)*S/float(K_EE),
            plasticity=plasticity,
            A=A,
            g=omega_g,
            o=omega_o,
            sigma_z=sigma_z)
        conn.J_I = -np.sqrt(K_EE)*synapse.E_w
    elif inhibition == 'sparse':
        omega_g = params.omega['g']
        omega_o = params.omega['o']
        inh = Population(N_I, tau=tau, phi=ReLU(g=g).phi)
        conn_IE = SparseConnectivity(
                source=exc,
                target=inh,
                p=c,
                seed=111,
                disable_pbar=True)
        conn_EI = SparseConnectivity(
                source=inh,
                target=exc,
                p=c,
                seed=112,
                disable_pbar=True)
        K_EE, K_IE, K_EI, K_II = conn.K, conn_IE.K, conn_EI.K, 0
        synapse = RectifiedSynapse(
            K_EE, K_IE, K_EI, K_II,
            alpha=(P-1)*S/float(K_EE),
            plasticity=plasticity,
            A=A,
            g=omega_g,
            o=omega_o,
            sigma_z=sigma_z)
        conn_IE.set_all(synapse.h_IE(1))
        conn_EI.set_all(synapse.h_EI(1))
    else:
        synapse = LinearSynapse(conn.K, A=A)

    conn.store_sequences(patterns, lambda x: x, plasticity.f, plasticity.g)
    
    errcode = 0
    state, overlaps, correlations = [], [], []
    W_sequence = np.copy(conn.W.data)
    W_pert = np.zeros_like(W_sequence) 
    conn_W = np.zeros((n_days,conn.W.data.size))
    rng = np.random.RandomState(seed=47)
    for n in range(0, n_days):
        z_n = rng.normal(scale=1, size=conn.W.data.size)
        if n == 0:
            W_pert = sigma_z*z_n
        else:
            W_pert = lambda_*W_pert + np.sqrt(1-lambda_**2)*sigma_z*z_n
        conn.W.data = synapse.h_EE(W_sequence + W_pert)
        conn_W[n,:] = conn.W.data
    
    return conn_W
