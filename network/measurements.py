import numpy as np
import pdb
import copy
from collections import defaultdict
from network import RateNetwork

class Measurement(object):
    def __init__(self, *args, **kwargs):
        pass

    def attach(self, learning_process):
        self.learning_process = learning_process


class Overlaps(Measurement):
    def __init__(self, t, sequences, patterns, plasticity, n=1, *args, **kwargs):
        super(Overlaps, self).__init__(self, *args, **kwargs)
        self.n = n
        self.t = t
        self.sequences = sequences
        self.patterns = patterns
        self.plasticity = plasticity
        self.data = defaultdict(lambda : np.array([]))

    def run(self, n):
        pop = self.learning_process.pop
        conn = self.learning_process.conn
        eps_r = self.learning_process.eps_r
        G = self.learning_process.G
        plasticity = self.plasticity
        conn_frozen = copy.deepcopy(conn)
        if eps_r > 0:
            for i in range(pop.size):
                conn_frozen.W[i] *= eps_r*G[i]
        net = RateNetwork(pop, c_EE=conn_frozen, formulation=1, disable_pbar=True)
        net.clear_state()
        net.simulate_euler(
            self.t,
            r0=pop.phi(self.patterns[0,0,:]),
            save_field=False)
        overlaps = self.sequences[0].overlaps(
            net,
            pop,
            phi=pop.phi,
            plasticity=plasticity,
            disable_pbar=True)
        self.data[n] = overlaps


class SequenceScore(Measurement):
    def __init__(self, mm_overlaps, n=1, *args, **kwargs):
        super(SequenceScore, self).__init__(self, *args, **kwargs)
        self.n = n
        self.data = defaultdict(lambda : np.array([]))
        self.mm_overlaps = mm_overlaps

    def run(self, n):
        overlaps = self.mm_overlaps.data[n]
        score1 = SequenceScore._score1(overlaps)
        score2 = SequenceScore._score1(overlaps)
        self.data[n] = {
            'center_of_mass': score1,
            'arg_max': score2
        }

    @staticmethod
    def _score1(overlaps):
        "Center of mass based score"
        P, T = overlaps.shape
        r_avg = overlaps.mean(axis=0)

        t_com = [int(np.sum(np.arange(T)*r)/r.sum()) for r in overlaps]
        r_com = np.vstack([
            np.roll(overlaps[i], shift=int(T/2)-t_com[i]) for i in range(P)])
        r_com_avg = r_com.mean(axis=0)
        r_diff_num = np.mean((r_com - r_com_avg)**2, axis=1)
        r_diff_den = np.mean((overlaps - r_avg)**2, axis=1)
        score = np.clip(1 - r_diff_num/r_diff_den, 0, np.inf)
        
        return score

    @staticmethod
    def _score2(overlaps):
        "Argmax based score (nanargmax, nanmean)"
        P, T = overlaps.shape
        r_avg = overlaps.mean(axis=0)

        t_argmax = [np.nanargmax(r) for r in overlaps]
        r_argmax = np.vstack([
            np.roll(overlaps[i], shift=int(T/2)-t_argmax[i]) for i in range(P)])
        r_argmax_avg = r_argmax.mean(axis=0)
        r_diff_num2 = np.nanmean((r_argmax - r_argmax_avg)**2, axis=1) 
        r_diff_den2 = np.nanmean((overlaps - r_avg)**2, axis=1)
        score = np.clip(1 - r_diff_num2/r_diff_den2, 0, np.inf)

        return score


class SparsityScore(Measurement):
    def __init__(self, mm_overlaps, n=1, *args, **kwargs):
        super(SparsityScore, self).__init__(self, *args, **kwargs)
        self.n = n
        self.data = defaultdict(lambda : np.array([]))
        self.mm_overlaps = mm_overlaps

    def run(self, n):
        overlaps = self.mm_overlaps.data[n]
        P, T = overlaps.shape

        # Spatial
        num = np.sqrt(P) - np.sum(np.abs(overlaps), axis=0) / \
              np.sqrt(np.sum(overlaps**2, axis=0))
        den = np.sqrt(P) - 1
        score1 = num/den
        mask_idxs = overlaps.sum(axis=0) < 3*overlaps.std()
        score1[mask_idxs] = 0

        # Temporal
        num2 = np.sqrt(T) - np.sum(np.abs(overlaps), axis=1) / \
               np.sqrt(np.sum(overlaps**2, axis=1))
        den2 = np.sqrt(T) - 1
        score2 = num2/den2

        self.data[n] = {
            'spatial': np.asarray(score1),
            'temporal': np.asarray(score2)
        }
        

class WeightGradient(Measurement):
    def __init__(self, conn, n=1, *args, **kwargs):
        super(WeightGradient, self).__init__(self, *args, **kwargs)
        self.n = n
        self.data = defaultdict(lambda : np.array([]))
        self.J_prev = np.zeros_like(conn.W.data)
        self.J_prev[:] = conn.W.data

    def run(self, n):
        conn = self.learning_process.conn
        grad = np.mean(np.abs((conn.W.data - self.J_prev)/2.))
        self.J_prev[:] = conn.W.data
        self.data[n] = np.r_[self.data[n], grad]