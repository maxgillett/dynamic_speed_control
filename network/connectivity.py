import pdb
import math
import logging
from numba import jit, njit, vectorize, prange, cuda
import numba
import numpy as np
import scipy.sparse
from tqdm.auto import trange
from scipy.stats import lognorm, norm
from itertools import cycle

import importlib 
cupy_loader = importlib.find_loader('cupy')
if cupy_loader is not None:
    import cupy as cp
    import cupyx as cpx
else:
    cp = np

logger = logging.getLogger(__name__)

class Connectivity(object):
    def __init__(self):
        self.W = None
        self.disable_pbar = False

    @property
    def size(self):
        return self.W.data.size

    def scale_all(self, value):
        self.W *= value

    def set_connectivity(self):
        raise NotImplementedError

    def reset_connectivity(self):
        raise NotImplementedError

    @staticmethod
    def _store_assocation(ji, inputs, f, g, k=1, p=1, disable_pbar=False):
        """
        inputs: S x P x N
        Store heteroassociative connectivity
        Inputs:
            ij: row index --> column indices
        """
        S, P, N  = inputs.shape
        row = []
        col = []
        data = []
        for n in trange(len(inputs), disable=disable_pbar):
            seq = inputs[n]
            for i in trange(seq.shape[1], disable=disable_pbar):
                j = ji[i]
                # $f(xi_i^{\mu+p}) * g(xi_j^{\mu})$
                if p == 0:
                    w = np.sum(f(seq[:,i][:,np.newaxis]) * g(seq[:,j]), axis=0) 
                else:
                    w = np.sum(f(seq[p:,i][:,np.newaxis]) * g(seq[:-p,j]), axis=0) 
                w *= k
                row.extend([i]*len(j))
                col.extend(j)
                data.extend(w)
        return data, row, col

    @staticmethod
    def _store_sequences(ij, inputs, f, g, disable_pbar=False):
        """
        inputs: S x P x N
        Store heteroassociative connectivity
        """
        S, P, N  = inputs.shape
        row = []
        col = []
        data = []
        for n in trange(len(inputs), disable=disable_pbar):
            seq = inputs[n]
            for j in trange(seq.shape[1], disable=disable_pbar):
                i = ij[j]
                # $f(xi_i^{\mu+1}) * g(xi_j^{\mu})$
                w = np.sum(f(seq[1:,i]) * g(seq[:-1,j][:,np.newaxis]), axis=0) 
                row.extend(i)
                col.extend([j]*len(i))
                data.extend(w)
        return data, row, col


    @staticmethod
    def _store_attractors(ij, inputs, f, g, disable_pbar=False):
        """
        inputs: P x N
        Store autoassociative connectivity
        """
        P, N  = inputs.shape
        row = []
        col = []
        data = []
        for j in trange(inputs.shape[1], disable=disable_pbar):
            i = ij[j]
            w = np.sum(f(inputs[:,i]) * g(inputs[:,j][:,np.newaxis]), axis=0) 
            row.extend(i)
            col.extend([j]*len(i))
            data.extend(w)
        return data, row, col

    @staticmethod
    def _set_all(ij, value):
        row = []
        col = []
        data = []
        for j in range(len(ij)):
            i = ij[j]
            row.extend(i)
            col.extend([j]*len(i))
            data.extend([value]*len(i))
        return data, row, col
                

class SparseConnectivity(Connectivity):
    def __init__(self, source, target, p=0.005, fixed_degree=False, seed=42, device="cpu",
            disable_pbar=False, sparse_mat_type='csr'):

        self.device = device
        self.sparse_mat_type = sparse_mat_type
        self.W = scipy.sparse.coo_matrix(
                (target.size, source.size), dtype=np.float32).asformat(sparse_mat_type)
        self.p = p
        self.K = p*target.size
        self.ij = []
        self.disable_pbar = disable_pbar
        logger.info("Building connections from %s to %s" % (source.name, target.name))
        if fixed_degree:
            n_neighbors = np.asarray([int(p*source.size)]*target.size)
        else:
            n_neighbors = np.random.RandomState(seed).binomial(
                source.size,
                p=p,
                size=target.size)

        # Build structural connectivity, and compute row/column index lookup tables
        # ji: list of column indices {j} for row i
        # ij: list of row indices {i} for column j

        @njit
        def build_ji(source_size, target_size, n_neighbors):
            np.random.seed(seed)
            ji = []
            for i in range(target_size): # Iterate over rows
                # Exclude self-connections
                arr = np.arange(source_size)
                arr2 = np.concatenate((arr[:i], arr[i+1:]))
                j_subset = np.random.choice(arr2, size=n_neighbors[i], replace=False)
                ji.append(sorted(j_subset))
            return ji
        self.ji = build_ji(source.size, target.size, n_neighbors) 

        self.ij = [[] for _ in range(source.size)]
        for row_idx, col_idxs in enumerate(self.ji):
            for col_idx in col_idxs:
                self.ij[col_idx].append(row_idx)

        self.a = a = np.cumsum([0]+[len(d) for d in self.ji])
        self.b = b = np.hstack(self.ji)
        self.state = np.array([], ndmin=2).reshape(sum([len(ij) for ij in self.ij]),0)

        if device == "cpu":
            if sparse_mat_type == "coo":
                # Optimized sparse matrix - dense vector product
                @njit(parallel=True)
                def dot(data, r_t):
                    y = np.zeros(r_t.size)
                    for i in prange(r_t.size):
                        j = b[a[i]:a[i+1]]
                        y[i] = np.sum(data[a[i]:a[i+1]] * r_t.take(j))
                    return y
                self.dot = dot
            elif sparse_mat_type == "csr":
                def dot(data, r_t):
                    return self.W.dot(r_t)
                self.dot = dot
            else:
                raise NotImplementedError
        elif device == "gpu":
            if sparse_mat_type == "coo":
                self.W = cpx.scipy.sparse.coo_matrix(self.W)
                def dot(data, r_t):
                    return self.W.dot(r_t)
                self.dot = dot
            elif sparse_mat_type == "csr":
                self.W = cpx.scipy.sparse.csr_matrix(self.W)
                def dot(data, r_t):
                    return self.W.dot(r_t)
                self.dot = dot
            else:
                raise NotImplementedError

    def store_associations(self,
            inputs,
            h=lambda x:x,
            f=lambda x:x,
            g=lambda x:x,
            k=1,
            p=1):
        """
        k: Weighting
        p: Associated pattern offset
        """
        N = inputs.shape[2]
        data, row, col = Connectivity._store_assocation(
                self.ji, inputs, f, g, k, p, self.disable_pbar)
        data = h(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.asformat(self.sparse_mat_type)

    def store_sequences(self,
            inputs,
            h=lambda x:x,
            f=lambda x:x,
            g=lambda x:x):
        N = inputs.shape[2]
        logger.info("Storing sequences")
        data, row, col = Connectivity._store_sequences(self.ij, inputs, f, g, self.disable_pbar)
        logger.info("Applying synaptic transfer function")
        #pdb.set_trace()
        data = h(data)
        logger.info("Building sparse matrix")
        if self.device == "cpu":
            W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        else:
            data = cp.asarray(data)
            row = cp.asarray(row)
            col = cp.asarray(col)
            W = cpx.scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.asformat(self.sparse_mat_type)

    def store_attractors(self,
            inputs,
            h=lambda x:x,
            f=lambda x:x,
            g=lambda x:x):
        logger.info("Storing attractors")
        data, row, col = Connectivity._store_attractors(self.ij, inputs, f, g, self.disable_pbar)
        data = h(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.asformat(self.sparse_mat_type)

    def set_random(self, var, h=lambda x:x):
        data, row, col = Connectivity._set_all(self.ij, 1)
        data = np.asarray(data, dtype=float)
        data[:] = np.sqrt(var)*np.random.randn(data.size)
        data = h(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.asformat(self.sparse_mat_type)

    def set_weights(self, data, row, col):
        "NOTE: Adds to, but does not overwrite existing weights"
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.asformat(self.sparse_mat_type)

    def set_all(self, value):
        "NOTE: Adds to, but does not overwrite existing weights"
        data, row, col = Connectivity._set_all(self.ij, value)
        if self.device == "cpu":
            W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        else:
            data = cp.asarray(data)
            row = cp.asarray(row)
            col = cp.asarray(col)
            W = cpx.scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.asformat(self.sparse_mat_type)


class DenseConnectivity(Connectivity):
    def __init__(self, source, target, seed=42, disable_pbar=False):
        self.disable_pbar = disable_pbar
        self.W = np.zeros((target.size, source.size), dtype=np.float32)
        self.K = target.size-1

    def store_sequences(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        """
        inputs: S x P x N
        """
        N = inputs.shape[2]
        logger.info("Storing sequences")
        for n in trange(len(inputs), disable=self.disable_pbar):
            seq = inputs[n]
            for mu in trange(seq.shape[0]-1, disable=self.disable_pbar):
                W = h(np.outer(f(seq[mu+1,:]), g(seq[mu,:])))
                diag = np.diagonal(W)
                diag.setflags(write=True)
                diag.fill(0)
                self.W += W

    def store_attractors(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        pass

    def set_weights(self, data, row, col):
        pass

    def set_all(self, value):
        pass



class PlasticityRule(object):
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get('device', 'cpu')


class BilinearPlasticityRule(PlasticityRule):
    def __init__(self, conn, synapse, *args, **kwargs):
        super(BilinearPlasticityRule, self).__init__(self)

        @vectorize
        def f(x):
            return x

        @vectorize
        def g(x):
            return x

        self.f = f
        self.g = g

        ij, a, b, = conn.ij, conn.a, conn.b
        K, A = synapse.K_EE, synapse.A

        @njit(parallel=True)
        def update(J_data, dJ_data,
                r_t, r_t_prev, r_t_prev2, 
                k0=0, k1=1, k2=0,
                dt=1e-3, tau_w=1):
            """
            Compute single in-place Euler update for plasticity rule
            Inputs: 
                J_data: Sparse COO weight matrix data.
                dJ_data: Weight update.
            """
        
            for i in prange(r_t.shape[0]):
                j = b[a[i]:a[i+1]]
                dw = np.zeros(j.size)
                if k0 != 0:
                    g_r_avg = np.mean(k0*g(r_t[j]))
                    dw += k0*g(r_t[j])-g_r_avg
                if k1 != 0:
                    g_r_avg = np.mean(k1*g(r_t_prev[j]))
                    dw += k1*g(r_t_prev[j])-g_r_avg
                if k2 != 0:
                    g_r_avg = np.mean(k2*g(r_t_prev2[j]))
                    dw += k2*g(r_t_prev2[j])-g_r_avg
                dw *= f(r_t.take(i))
                dJ_data[slice(a[i],a[i+1])] = dw
            J_data[:] = J_data*(1-dt/tau_w) + dt*A*dJ_data/K
        self.update = update


class ThresholdPlasticityRule(PlasticityRule):
    def __init__(self,
            x_f,
            q_f,
            x_g=None,
            q_g=None,
            rv=scipy.stats.norm,
            device='cpu',
            *args,
            **kwargs):

        super(ThresholdPlasticityRule, self).__init__(*args, **kwargs)

        self.device = device
        if not x_g: x_g = x_f
        if not q_g: q_g = rv.cdf(x_g)
        self.x_f, self.q_f = x_f, q_f
        self.x_g, self.q_g = x_g, q_g

        self.build_plasticity_functions()

    def build_plasticity_functions(self):
        device = self.device
        x_f, q_f = self.x_f, self.q_f
        x_g, q_g = self.x_g, self.q_g

        if device == "cpu":
            @vectorize
            def f(x):
                if x > x_f:
                    return q_f
                else:
                    return -(1-q_f)
            @vectorize
            def g(x):
                if x > x_g:
                    return q_g
                else:
                    return -(1-q_g)
            self.f = f
            self.g = g
        elif device == "gpu":
            # CuPy
            def f(x):
                return cp.where(x > x_f, q_f, -(1-q_f))
            def g(x):
                return cp.where(x > x_g, q_g, -(1-q_g))
            self.f = f
            self.g = g

            # Numba
            x_f_ = float(x_f)
            x_g_ = float(x_g)
            @cuda.jit(device=True)
            def f(x):
                if x > x_f_:
                    return q_f
                else:
                    return -(1-q_f)
            @cuda.jit(device=True)
            def g(x):
                if x > x_g_:
                    return q_g
                else:
                    return -(1-q_g)
            self.f_numba = f
            self.g_numba = g


class TriphasicPlasticityRule(ThresholdPlasticityRule):
    def __init__(self, conn, synapse, *args, **kwargs):
        super(TriphasicPlasticityRule, self).__init__(*args, **kwargs)

        f, g = self.f, self.g
        ij, a, b, = conn.ij, conn.a, conn.b
        K, A = synapse.K_EE, synapse.A

        # Subtract mean
        zero_avg = kwargs.get("zero_avg", True)

        @njit(parallel=True)
        def update(J_data, dJ_data,
                r_t, r_t_prev, r_t_prev2, 
                k0=0, k1=1, k2=0,
                dt=1e-3, tau_w=1):
            """
            Compute single in-place Euler update for plasticity rule
            Inputs: 
                J_data: Sparse COO weight matrix data.
                dJ_data: Weight update.
            """
        
            for i in prange(r_t.shape[0]):
                j = b[a[i]:a[i+1]]
                dw = np.zeros(j.size)
                if k0 != 0:
                    if zero_avg:
                        g_r_avg = np.mean(k0*g(r_t[j]))
                    else:
                        g_r_avg = 0
                    dw += k0*g(r_t[j])-g_r_avg
                if k1 != 0:
                    if zero_avg:
                        g_r_avg = np.mean(k1*g(r_t_prev[j]))
                    else:
                        g_r_avg = 0
                    dw += k1*g(r_t_prev[j])-g_r_avg
                if k2 != 0:
                    if zero_avg:
                        g_r_avg = np.mean(k2*g(r_t_prev2[j]))
                    else:
                        g_r_avg = 0
                    dw += k2*g(r_t_prev2[j])-g_r_avg
                dw *= f(r_t.take(i))
                dJ_data[slice(a[i],a[i+1])] = dw
            J_data[:] = J_data*(1-dt/tau_w) + dt*A*dJ_data/K
        self.update = update


class GeneralizedHebbianRule(ThresholdPlasticityRule):
    def __init__(self, conn, synapse, *args, **kwargs):
        super(GeneralizedHebbianRule, self).__init__(*args, **kwargs)
        self.device = kwargs.get('device', 'cpu')
        self.K, self.A = synapse.K_EE, synapse.A
        self.build_update(conn) #, synapse, *args, **kwargs)

    def build_update(self, conn): #, synapse, *args, **kwargs):
        ij, a, b, = conn.ij, conn.a, conn.b
        K, A = self.K, self.A

        if self.device == "cpu":
            f, g = self.f, self.g

            @njit(parallel=True)
            def update(J_data, dJ_data, lambda_pre, lambda_post, r_mat, Delta_t=1, dt=1e-3):
                """
                J_data: data for underlying sparse matrix in COO format
                dJ_data: data buffer for weight update in COO format
                lambda_pre: 1d array of temporal offset weights
                r_mat: 2d array of firing rates, shape: (len(lambda_pre), N)
                """
                for i in prange(r_mat.shape[1]):
                    j = b[a[i]:a[i+1]]

                    # Pre + symmetric components
                    dw_pre = np.zeros(j.size)
                    for k in range(lambda_pre.size):
                        lambda_k = lambda_pre[k]
                        if lambda_k != 0:
                            r_t = r_mat[k*Delta_t]
                            g_r_avg = np.mean(lambda_k*g(r_t[j]))
                            dw_pre += lambda_k*g(r_t[j])-g_r_avg
                    dw_pre *= f(r_mat[0].take(i))

                    # Post components
                    if lambda_post.size > 0:
                        dw_post = np.zeros(j.size)
                        for k in range(1,lambda_post.size):
                            lambda_k = lambda_post[k]
                            if lambda_k != 0:
                                r_t = r_mat[k*Delta_t]
                                dw_post += lambda_k*f(r_t.take(i))
                        g_r_avg = np.mean(g(r_mat[0][j]))
                        dw_post *= (g(r_mat[0][j]) - g_r_avg)
                        dJ_data[slice(a[i],a[i+1])] = dw_pre + dw_post
                    else:
                        dJ_data[slice(a[i],a[i+1])] = dw_pre

                J_data[:] = J_data + dt*A*dJ_data/K

            def update_v2(J_data, dJ_data, lambda_pre, lambda_post, r_mat, Delta_t=1, dt=1e-3):
                raise NotImplementedError

        elif self.device == "gpu":

            f, g = self.f_numba, self.g_numba
            b_ = cuda.to_device(b.reshape(-1,int(K)))

            @cuda.jit
            def kernel(dJ_data, lambda_pre, lambda_post, r_mat_0, r_mat_take, Delta_t, g_r_avg):
                i,j = cuda.grid(2)
                if i < r_mat_0.size:
                    if j < 200: # FIXME Do not hardocde K=200
                        dw = 0
                        for k in range(lambda_pre.size):
                            lambda_k = lambda_pre[k]
                            if lambda_k != 0:
                                dw += lambda_k*g(r_mat_take[k,i,j]) - \
                                      lambda_k*g_r_avg[k,i]
                        dw *= f(r_mat_0[i])
                        dJ_data[i*200+j] = dw # FIXME Do not hardocde K=200

            def update(J_data, dJ_data, lambda_pre, lambda_post, r_mat, Delta_t=1, dt=1e-3):
                c = r_mat[::Delta_t,:].T.take(np.asarray(b_), axis=0)
                r_mat_take = cp.rollaxis(c, axis=2, start=0)
                g_r_avg = cp.mean(self.g(r_mat_take), axis=2)
                threadsperblock = (16,16)#128
                blockspergrid_x = math.ceil(b_.shape[0] / threadsperblock[0])
                blockspergrid_y = math.ceil(b_.shape[1] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                kernel[blockspergrid, threadsperblock](
                        dJ_data,
                        lambda_pre,
                        lambda_post,
                        r_mat[0],
                        r_mat_take,
                        Delta_t,
                        g_r_avg)
                J_data[:] = J_data + dt*A*cp.asarray(dJ_data)/K

            def update_v2(J_data, dJ_data, lambda_pre, lambda_post, r_mat, v_mat, Delta_t=1, dt=1e-3):
                c = r_mat[::Delta_t,:].T.take(np.asarray(b_), axis=0)
                r_mat_take = cp.rollaxis(c, axis=2, start=0)
                g_r_avg = cp.mean(self.g(r_mat_take), axis=2)
                threadsperblock = (16,16)#128
                blockspergrid_x = math.ceil(b_.shape[0] / threadsperblock[0])
                blockspergrid_y = math.ceil(b_.shape[1] / threadsperblock[1])
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                kernel[blockspergrid, threadsperblock](
                        dJ_data,
                        lambda_pre,
                        lambda_post,
                        r_mat[0],
                        r_mat_take,
                        Delta_t,
                        g_r_avg)
                J_data[:] = J_data + dt*A*cp.asarray(dJ_data)/K
                    
        self.update = update

        self.update_v2 = update

class RescalingPlasticityRule(PlasticityRule):
    def __init__(self, conn, zero_mean=False, *args, **kwargs):
        super(RescalingPlasticityRule, self).__init__(*args, **kwargs)
        self.device = kwargs.get('device', 'cpu')

        a, b, N = conn.a, conn.b, len(conn.ji) 

        if self.device == "cpu":
            @njit(parallel=True)
            def update(data, std_w):
                """
                Rescale rows of sparse data matrix to std_w
                Inputs:
                    data: Sparse COO weight matrix data.
                    std_w: Desired row-wise standard deviation of data.
                    zero_mean: Whether or not to zero the mean. We assume that the mean 
                        is already zeroed, and therefore set the default value to false.
                """
                for i in prange(N):
                    J_mean_2 = np.mean(data[a[i]:a[i+1]]**2)
                    if J_mean_2 > 0:
                        if zero_mean:
                            J_mean = np.mean(data[a[i]:a[i+1]])
                            data[a[i]:a[i+1]] -= J_mean
                        data[a[i]:a[i+1]] /= np.sqrt(J_mean_2)
                        data[a[i]:a[i+1]] *= std_w

        elif self.device == "gpu":
            def update(data, std_w):
                for i in range(N):
                    J_mean_2 = cp.mean(data[a[i]:a[i+1]]**2)
                    data[a[i]:a[i+1]] /= cp.sqrt(J_mean_2)
                    data[a[i]:a[i+1]] *= std_w
        
        self.update = update


class ConstantSumPlasticityRule(PlasticityRule):
    def __init__(self, conn, *args, **kwargs):
        super(ConstantSumPlasticityRule, self).__init__(*args, **kwargs)

        a, b, N = conn.a, conn.b, len(conn.ji) 
        J_row_sum_0 = np.squeeze(np.asarray(np.abs(conn.W).mean(axis=1)))

        if conn.sparse_mat_type == "coo":
            @njit(parallel=True)
            def update(data, std_w):
                """
                Inputs:
                    data: Sparse COO weight matrix data.
                    std_w: Desired row-wise standard deviation of data.
                """
                for i in prange(N):
                    J_mean = (np.mean(np.abs(data[a[i]:a[i+1]])) - J_row_sum_0[i])
                    data[a[i]:a[i+1]] -= J_mean

        elif conn.sparse_mat_type == "csr":
            @njit(parallel=True)
            def update(data, std_w):
                pass

        self.update = update

class MultiplicativeHomeostaticRule(PlasticityRule):
    def __init__(self, conn, zero_mean=False, *args, **kwargs):
        super(MultiplicativeHomeostaticRule, self).__init__(*args, **kwargs)

        a, b, N = conn.a, conn.b, len(conn.ji) 

        @njit(parallel=True)
        def update(Jdata, Wdata, H, r_t, r0, tau_H, H_min=0, H_max=1, dt=1e-3):
            for i in prange(N):
                #j = b[a[i]:a[i+1]] 
                H[i] = H[i] + dt/tau_H * (1-r_t[i]/r0) * H[i]
                #if H[i] < H_min:
                #    H[i] = H_min
                if H[i] > H_max:
                    H[i] = H_max
                Wdata[a[i]:a[i+1]] = Jdata[a[i]:a[i+1]] * H[i]
        self.update = update


class SynapticTransferFunction(object):
    def __init__(self, K):
        pass

class LinearSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, A):
        super(LinearSynapse, self).__init__(self)
        self.A = A
        self.K_EE = K_EE

    def h_EE(self, J):
        return self.A * np.asarray(J) / self.K_EE

    def h(self, J):
        return self.h_EE(J)


class RectifiedSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, K_IE, K_EI, K_II, alpha, plasticity, A=1, g=1., o=0, sigma_z=0):
        super(RectifiedSynapse, self).__init__(self)

        @np.vectorize
        def rectify(x):
          if x<0:
              return 0
          else:
              return x

        self.K_EE, self.K_IE, self.K_EI, self.II = K_EE, K_IE, K_EI, K_II
        f_fun, x_f, q_f = plasticity.f, plasticity.x_f, plasticity.q_f
        g_fun, x_g, q_g = plasticity.g, plasticity.x_g, plasticity.q_g
        self.A, self.g, self.o, = A, g, o

        gamma = norm.expect(lambda x: f_fun(x)**2) * \
                norm.expect(lambda x: g_fun(x)**2)
        self.E_w = norm.expect(
                lambda x: rectify(g*(A*np.sqrt(alpha*gamma)+sigma_z)*x + o),
                scale=1)
        self.E_w_2 = norm.expect(
                lambda x: rectify(g*(A*np.sqrt(alpha*gamma)+sigma_z)*x + o)**2,
                scale=1)
    
    def h_EE(self, J):
        return self.A * (self.g * (J / np.sqrt(self.K_EE)) + self.o).clip(min=0) / np.sqrt(self.K_EE)

    def h_IE(self, J):
        return J * self.E_w / self.K_IE

    def h_EI(self, J):
        return -J * 1. / np.sqrt(self.K_EI)

    def h_II(self, J):
        return -J


class ExponentialSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, K_IE, K_EI, K_II, alpha, plasticity, A=1, g=1., o=0):
        super(ExponentialSynapse, self).__init__(self)

        @np.vectorize
        def exp(x):
          return np.exp(x)

        self.K_EE, self.K_IE, self.K_EI, self.II = K_EE, K_IE, K_EI, K_II
        f_fun, x_f, q_f = plasticity.f, plasticity.x_f, plasticity.q_f
        g_fun, x_g, q_g = plasticity.g, plasticity.x_g, plasticity.q_g
        self.A, self.g, self.o, = A, g, o

        gamma = norm.expect(lambda x: f_fun(x)**2) * \
                norm.expect(lambda x: g_fun(x)**2)
        self.E_w = norm.expect(
                lambda x: exp(g*A*np.sqrt(alpha*gamma)*x + o),
                scale=1)
        self.E_w_2 = norm.expect(
                lambda x: exp(g*A*np.sqrt(alpha*gamma)*x + o)**2,
                scale=1)
    
    def h_EE(self, J):
        return self.A * (self.g * (J / np.sqrt(self.K_EE)) + self.o).clip(min=0) / np.sqrt(self.K_EE)

    def h_IE(self, J):
        return J * self.E_w / self.K_IE

    def h_EI(self, J):
        return -J * 1. / np.sqrt(self.K_EI)

    def h_II(self, J):
        return -J
