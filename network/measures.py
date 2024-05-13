import pdb
import os
from itertools import product
from tqdm.auto import tqdm
import numpy as np
from connectivity import Connectivity
from pathos.pools import ProcessPool

class DataStore(object):
    def __init__(self, dirname):
        self.directory = dirname
        self.data_arr = []
        self.data = dict()
        self.keyvals = dict()

    def load(self, parallel=False, excluded_keys=[]):
        "Load the raw data files in specified directory"
        if parallel:
            filepaths = []
            for file in tqdm(os.listdir(self.directory)[:]):
                if file.endswith(".npy"):
                    filepaths.append(os.path.join(self.directory, file))
            func = lambda x: np.load(open(x, 'rb'), allow_pickle=True)
            with ProcessPool(len(filepaths)) as pool:
                for d in pool.map(func, filepaths):
                    self.data_arr.append(d.item())
        else:
            for file in tqdm(os.listdir(self.directory)[:]):
                if file.endswith(".npy"):
                    filepath = os.path.join(self.directory, file)
                    d = np.load(open(filepath, 'rb'), allow_pickle=True).item()
                    for key in excluded_keys:
                        if key in d.keys():
                            del d[key]
                    self.data_arr.append(d)

    def process(self, keys):
        "Convert raw data files to dict indexed by keys"
        data = dict()
        keyvals = dict(zip(keys,[set() for _ in range(len(keys))]))
        for d in self.data_arr:
            vals = [d['params'][key] for key in keys]
            for key, val in zip(keys, vals):
                keyvals[key].add(val)
            data[tuple(vals)] = d
        for key, val in keyvals.items():
            keyvals[key] = sorted(list(val))
        self.data = data
        self.keyvals = keyvals

    def slice(self, measurement, output, keys, raise_on_error=False):
        "Generate (multidimesional) slice through a particular measurement"
        for idxs, vals in zip(
                product(*[range(len(self.keyvals[key])) for key in keys]),
                product(*[self.keyvals[key] for key in keys])):
            try:
                output[idxs] = self.data[vals]['measurements'][measurement].data
            except:
                if raise_on_error:
                    raise StandardError
    

class Measurement(object):
    def __init__(self, n, *args, **kwargs):
        self.t = []
        self.data = []
        self.epsilon_c = kwargs.get('epsilon_c', None)
    
    def attach(self, learning_process):
        self.lp = learning_process
        if self.epsilon_c is None:
            self.epsilon_c = self.lp.epsilon_r
        
class Overlaps(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(Overlaps, self).__init__(self, *args, **kwargs)
        self.name = kwargs.get('name', 'overlaps')
        self.n = n
        self.simulation_required = True
        self.epsilon_c = kwargs.get('epsilon_c', None)
        
    def run(self, n, overlaps, correlations):
        self.t.append(n)
        self.data.append(overlaps)

class Correlations(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(Correlations, self).__init__(self, *args, **kwargs)
        self.name = kwargs.get('name', 'correlations')
        self.n = n
        self.simulation_required = True
        self.epsilon_c = kwargs.get('epsilon_c', None)
        
    def run(self, n, overlaps, correlations):
        self.t.append(n)
        self.data.append(correlations)
        
class FiringRates(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(FiringRates, self).__init__(self, *args, **kwargs)
        self.name = "firing_rates"
        self.n = n
        self.simulation_required = True
        
    def run(self, n, overlaps, correlations):
        self.t.append(n)
        state = self.lp.pop.state
        self.data.append(state)
    
        
class FullConnectivity(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(FullConnectivity, self).__init__(self, *args, **kwargs)
        self.name = "full_connectivity"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        self.t.append(n)
        self.data.append(self.lp.conn.W.data)
        
class SequenceScore(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(SequenceScore, self).__init__(self, *args, **kwargs)
        self.name = kwargs.get('name', 'sequence_score')
        self.n = n
        self.simulation_required = True
        self.epsilon_c = kwargs.get('epsilon_c', None)
        self.debug = kwargs.get('debug', False)
        
    def run(self, n, overlaps, correlations):
        P, T = overlaps.shape
        r_avg = overlaps.mean(axis=0)
        t_argmax = [np.nanargmax(r) for r in overlaps]
        r_argmax = np.vstack([
            np.roll(overlaps[i], shift=int(T/2)-t_argmax[i]) for i in range(P)
        ])
        r_argmax_avg = r_argmax.mean(axis=0)
        r_diff_num2 = np.nanmean((r_argmax - r_argmax_avg)**2, axis=1)
        r_diff_den2 = np.nanmean((overlaps - r_avg)**2, axis=1)
        score = np.clip(1 - r_diff_num2/r_diff_den2, 0, np.inf)
        self.t.append(n)
        self.data.append(score)
        if self.debug:
            print(self.name, n, score.mean())

class SequenceScoreNeurons(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(SequenceScoreNeurons, self).__init__(self, *args, **kwargs)
        self.name = "sequence_score_neurons"
        self.n = n
        self.simulation_required = True
        
    def run(self, n, overlaps, correlations):
        P, T = overlaps.shape
        r_avg = overlaps.mean(axis=0)
        t_argmax = [np.nanargmax(r) for r in overlaps]
        r_argmax = np.vstack([
            np.roll(overlaps[i], shift=int(T/2)-t_argmax[i]) for i in range(P)
        ])
        r_argmax_avg = r_argmax.mean(axis=0)
        r_diff_num2 = np.nanmean((r_argmax - r_argmax_avg)**2, axis=1)
        r_diff_den2 = np.nanmean((overlaps - r_avg)**2, axis=1)
        score = np.clip(1 - r_diff_num2/r_diff_den2, 0, np.inf)
        self.t.append(n)
        self.data.append(score)

class WeightMatrix(Measurement):
    def __init__(self, n=1, conn=Connectivity(), *args, **kwargs):
        super(WeightMatrix, self).__init__(self, n, *args, **kwargs)
        self.name = "weight_matrix"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        conn = self.lp.conn
        W = conn.W.tocoo()
        mat = np.vstack([W.row, W.col, W.data])
        self.t.append(n)
        self.data.append(mat)
        
class WeightGradient(Measurement):
    def __init__(self, n=1, conn=Connectivity(), *args, **kwargs):
        super(WeightGradient, self).__init__(self, n, *args, **kwargs)
        self.name = "weight_gradient"
        self.n = n
        self.J_prev = conn.W.data.copy()
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        conn = self.lp.conn
        grad = np.mean(np.abs((conn.W.data - self.J_prev)/2.))
        self.J_prev[:] = conn.W.data.copy()
        self.t.append(n)
        self.data.append(grad)
        
class WeightSubset(Measurement):
    def __init__(self, n=1, m=100, conn=Connectivity(), *args, **kwargs):
        super(WeightSubset, self).__init__(self, n, *args, **kwargs)
        self.name = "weight_subset"
        self.n = n
        self.idxs = np.random.choice(range(conn.W.data.size), m, replace=False)
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        conn = self.lp.conn
        device = self.lp.device
        self.t.append(n)
        if device == "cpu":
            self.data.append(conn.W.data[self.idxs])
        elif device == "gpu":
            self.data.append(conn.W.data.get()[self.idxs])

class LambdaVector(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(LambdaVector, self).__init__(self, n, *args, **kwargs)
        self.name = "lambda_vector"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        lambda_vec = kwargs.get('lambda_vec', None)
        self.t.append(n)
        self.data.append(lambda_vec.copy())

class LambdaPre(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(LambdaPre, self).__init__(self, n, *args, **kwargs)
        self.name = "lambda_pre"
        self.n = n
        self.simulation_required = False
        self.debug = kwargs.get('debug', False)
        
    def run(self, n, *args, **kwargs):
        lambda_pre = kwargs.get('lambda_pre', None)
        self.t.append(n)
        self.data.append(lambda_pre.copy())
        if self.debug:
            print(self.name, n, lambda_pre.copy()[:3])

class LambdaPost(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(LambdaPost, self).__init__(self, n, *args, **kwargs)
        self.name = "lambda_post"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        lambda_post = kwargs.get('lambda_post', None)
        self.t.append(n)
        self.data.append(lambda_post.copy())

class FieldOverlap(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(FieldOverlap, self).__init__(self, n, *args, **kwargs)
        self.name = "field_overlap"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        patterns = self.lp.patterns
        N = self.lp.N
        P = patterns.shape[1]
        phi = self.lp.phi
        conn = self.lp.conn
        q = np.zeros((P,P))
        for mu in range(P):
            phi_xi = phi(patterns[0][mu])
            J_phi_xi = conn.dot(conn.W.data, phi_xi)
            for mu_prime in range(P):
                q[mu, mu_prime] = patterns[0][mu_prime].dot(J_phi_xi)/N
        self.t.append(n)
        self.data.append(q)
        
class MultiplicativeVariable(Measurement):
    def __init__(self, n=1, m=100, conn=Connectivity(), *args, **kwargs):
        super(MultiplicativeVariable, self).__init__(self, n, *args, **kwargs)
        self.name = "multiplicative_variable"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        G = kwargs.get('G', None)
        self.t.append(n)
        self.data.append(G.copy())
        
class PopulationRateMean(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(PopulationRateMean, self).__init__(self, n, *args, **kwargs)
        self.name = "population_rate_mean"
        self.n = n
        self.simulation_required = True
        
    def run(self, n, overlaps, correlations):
        i = overlaps[-1,:].argmax()
        r_mean = self.lp.pop.state[:,:i].mean()
        self.t.append(n)
        self.data.append(r_mean)
        
class PopulationRateVariance(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(PopulationRateVariance, self).__init__(self, n, *args, **kwargs)
        self.name = "population_rate_variance"
        self.n = n
        self.simulation_required = True
        
    def run(self, n, overlaps, correlations):
        i = overlaps[-1,:].argmax()
        r_var = self.lp.pop.state[:,:i].var()
        self.t.append(n)
        self.data.append(r_var)

class LearningState1(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(LearningState1, self).__init__(self, n, *args, **kwargs)
        self.name = "learning_state_1"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        h_mat = kwargs.get('h_mat', None)
        self.t.append(n)
        self.data.append(h_mat.copy())
        
class LearningState2(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(LearningState2, self).__init__(self, n, *args, **kwargs)
        self.name = "learning_state_2"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        v_mat = kwargs.get('v_mat', None)
        self.t.append(n)
        self.data.append(v_mat.copy())

class LearningState3(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(LearningState3, self).__init__(self, n, *args, **kwargs)
        self.name = "learning_state_3"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        r_mat = kwargs.get('r_mat', None)
        self.t.append(n)
        self.data.append(r_mat.copy())


class RateOverlaps(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(RateOverlaps, self).__init__(self, n, *args, **kwargs)
        self.name = "rate_overlaps"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        r_mat = kwargs.get('r_mat', None)
        Delta_t = self.lp.Delta_t
        patterns = self.lp.patterns
        N = self.lp.N
        P = patterns.shape[1]
        phi = self.lp.phi
        conn = self.lp.conn
        q = np.zeros((P,P))
        for mu in range(P):
            r = r_mat[Delta_t*mu,:] 
            for mu_prime in range(P):
                q[mu, mu_prime] = patterns[0][mu_prime].dot(r)/N
        self.t.append(n)
        self.data.append(q)


class ExternalInput(Measurement):
    def __init__(self, n=1, *args, **kwargs):
        super(ExternalInput, self).__init__(self, n, *args, **kwargs)
        self.name = "external_input"
        self.n = n
        self.simulation_required = False
        
    def run(self, n, *args, **kwargs):
        I_ext = self.lp.I_ext
        self.t.append(n)
        self.data.append(I_ext.get().copy())
