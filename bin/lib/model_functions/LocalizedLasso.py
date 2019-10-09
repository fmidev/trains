import sys
import numpy as np
import multiprocessing as mp
import concurrent.futures

from numpy.matlib import repmat
from math import ceil

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import kron, eye, csc_matrix, vstack, issparse
from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

import networkx as nx

from memory_profiler import profile

#class NetworkLasso(BaseEstimator):
class LocalizedLasso(BaseEstimator):
    """
    Network Lasso with primal-dual update
    Paper:  https://arxiv.org/abs/1903.11178
    """

    iteration_ = 0
    running_average_ = None
    coef_ = None

    def __init__(self, num_iter=10, lambda_net=1, lambda_exc=0.01, biasflag=0, R=None, clipping_threshold=None, batch_size=None, n_jobs=-1):
        """
            num_iter: int
               The number of iteration for iterative-least squares update: (default: 10)
            R : n x n dimensional matrix
               connection between rows


            """

        self.num_iter   = num_iter
        self.lambda_net = lambda_net
        self.lambda_exc = lambda_exc
        self.biasflag = biasflag
        self.R = R
        self.batch_size = batch_size
        self.n_jobs=n_jobs

        if clipping_threshold is None:
            self.clipping_threshold = 1/7
        else:
            self.clipping_threshold = clipping_threshold

    def fetch_connections(self, data, stations_to_pick=None):
        """
        Generate graph from given data

        data : DataFrame
               dataframe with columns 'src' and 'dst'
        """

        connections = {}
        all_stations = []

        def connect(src, dst):
            if src in connections:
                if dst not in connections[src]:
                    connections[src].append(dst)
            else:
                connections[src] = [dst]

            if src not in all_stations:
                all_stations.append(src)
            if dst not in all_stations:
                all_stations.append(dst)

        def connect_picked(src, dst):
            if src in connections and src in stations_to_pick:
                if dst not in connections[src] and dst in stations_to_pick:
                    connections[src].append(dst)
            elif src in stations_to_pick and dst in stations_to_pick:
                connections[src] = [dst]

            if src not in all_stations and src in stations_to_pick:
                all_stations.append(src)
            if dst not in all_stations and dst in stations_to_pick:
                all_stations.append(dst)

        if stations_to_pick is not None:
            data.apply(lambda x: connect_picked(x['src'], x['dst']), axis=1)
        else:
            data.apply(lambda x: connect(x['src'], x['dst']), axis=1)

        self.connections = connections
        self.all_stations = all_stations
        return connections

    def make_r(self, stations):
        """
        Make graph matrix based on data and connection info

        Returns array defining how rows are connected to each others based
        on train stations

        stations : lst (n,)
                   list of train station of each row in X

        return ndarray [num_rows, num_rows]
        """
        print('Making connection matrix R...')
        self.R = None
        station_index = []
        for row_station in stations:
            row = []
            for conn_station in stations:
                # Train station may appear only as end station
                if row_station not in self.connections:
                    row.append(0)
                # If row_station and conn_station are connected, add connection
                elif conn_station in self.connections[row_station]:
                    row.append(1)
                else:
                    row.append(0)

            r_csc = csc_matrix(row)
            self.R = vstack([self.R, r_csc])
            station_index.append(row_station)

        #self.R = np.array(R)
        self.station_index = station_index

        return self.R

    def make_r_dense(self, stations, allow_self_connections=True):
        """
        Make graph matrix based on data and connection info

        Returns array defining how rows are connected to each others based
        on train stations

        stations : lst (n,)
                   list of train station of each row in X

        TODO: should this made to undirected by R = R + R.T ?
        TODO should connections to self removed by R = R - np.eye(R.shape) ?

        return ndarray [num_rows, num_rows]
        """
        print('Making dense connection matrix R...')

        R = []
        station_index = []
        for row_station in stations:
            row = []
            for conn_station in stations:
                # Train station may appear only as end station
                if row_station not in self.connections:
                    row.append(0)
                # If row_station and conn_station are connected, add connection
                elif conn_station in self.connections[row_station]:
                    row.append(1)
                else:
                    row.append(0)
            R.append(row)
            station_index.append(row_station)

        R = np.array(R)
        n, d = R.shape
        # Make undirected and remove self connections
        if allow_self_connections:
            self.R = (R + R.T)
        else:
            self.R = (R + R.T) - np.eye(n,d)

        self.station_index = station_index

        return self.R

    def set_graph(self, graph_matrix):
        """
        Set graph information matrix

        graph_matrix : array (n,n)
        """
        self.R = graph_matrix

    def get_R(self, stations=None):
        """
        Return R, generate it if necessary
        """
        if self.R is None:
            self.R = self.make_r_dense(stations)

        return self.R

    def partial_fit(self, X, y, R=None, stations=None):
        """
        Partial fit
        """
        # If R is not given, assume correct R to be stored via set_graph
        if R is None:
            R = self.make_r_dense(stations)

        if self.batch_size is not None and self.batch_size < len(X):
            self.train_in_batches(X, y, R)
        else:
            self.batch_count = 1
            coef, it = fit_primal_dual(X,
                                       y,
                                       R,
                                       self.num_iter,
                                       self.clipping_threshold,
                                       self.iteration_,
                                       self.coef_)

            self.coef_ = coef
            self.iteration_ = it

    def train_in_batches(self, X, y, R):
        """ Train in batches """
        start = 0
        end = self.batch_size
        n, d = X.shape
        # For reporting
        self.batch_count = ceil(n/self.batch_size)

        print('Training in {} batches (size: {})'.format(self.batch_count, self.batch_size))

        process_count = 1
        if self.n_jobs < 1:
            process_count = min(mp.cpu_count(), self.batch_count)
        else:
            process_count = min(self.n_jobs, self.batch_count)

        futures = []
        coeffs = []
        with concurrent.futures.ProcessPoolExecutor(process_count) as executor:
            i = 1
            results = []
            while start < n:
                print('Running batch {}/{}'.format(i, self.batch_count))
                x_batch = X[start:end,:]
                y_batch = y[start:end]
                r_batch = R[start:end,start:end]
                futures.append(executor.submit(fit_primal_dual, x_batch, y_batch, r_batch, self.num_iter, self.clipping_threshold))

                start += self.batch_size
                end += self.batch_size
                i+=1

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                coeffs.append(res[0])
                self.iteration_ += res[1]

        self.coef_ = np.array(coeffs).mean(axis=0)


    #@profile
    def fit(self, X, y, stations = None):
        """
        Fit the model

        X : ndarray [n x d]
            features
        y : ndarray or list [n x 1]
            lables
        stations : lst [n,]
                   list of train station of each row in X and y
        """

        # If not set, attemp to make connections how rows are connected to stations
        R = self.get_R(stations)

        if self.batch_size is None or self.batch_size > len(X):
            self.batch_count = 1
            coef, i = fit_primal_dual(X,
                                      y,
                                      R,
                                      self.num_iter, self.clipping_threshold)
            self.coef_ = coef
            self.iteration_ += i
        else:
            self.train_in_batches(X, y, R)

    def predict(self, X, _=None):
        """
        Predict

        X : ndarray [n x d]
            features

        return y_pred (lst [n,1]), weights (ndarray [n,d])
        """

        return self.predict_primal_dual(X, self.coef_), self.coef_

    def predict_primal_dual(self, X, W, debug=False):
        """
        Predict

        This method assumes that training is done with primal-dual method

        X : array (n, d)
          : features

        return array(n, 1)
        """
        # Traditional prediction for unseen data
        if debug:
            print(X)
            print(W)
            y_pred = X@W
            print(y_pred)

        return X@W

        # To predict missing labels
        #return np.sum(X * W, axis=1)


def pm_iteration(w_hat, gamma_block, D_block, u_hat, X_sampling_idx, y_sampling, X_sampling, d, n, gamma_vec, feature_block_masked, feature_norm_squared_vec, lambda_block, nro_edges, clipping_threshold, it):
    # alg. 2
    w_new = w_hat - .5 * gamma_block @ (D_block.transpose() @ u_hat)
    # alg. 3, eq. 19
    w_new = block_thresholding(w_new,
                               X_sampling_idx,
                               y_sampling,
                               X_sampling,
                               d,
                               n,
                               gamma_vec,
                               feature_block_masked,
                               feature_norm_squared_vec
                               ) # checked

    w_tilde = 2 * w_new - w_hat
    w_hat = w_new

    # alg. 4
    u_new = u_hat + .5 * lambda_block @ (D_block @ w_tilde)
    # alg. 5
    u_hat = block_clipping(u_new, d, nro_edges, clipping_threshold) # checked

    running_average = (running_average * it + w_hat) / (it+1)

    return running_average


def block_thresholding(W_in, X_idx, y, X, d, n, gamma_vec, feature_block_masked, feature_norm_squared_vec):
    """
    Eq. 49 (?) --> solution of eq. 41

    W_in : array (n*d, 1)
           weights in
    X_idx : list
            list of sampling set indices in X
    y : list
        labels (same length than X_idx)
    X : array (n, d)
        features
    d : int
        feature dimensions
    n : int
        number of nodes
    gamma_vec : lst or array (n,1)
              : limits for thresholding
    feature_block_masked : array (9500 x 9500)
                         : masked block feature matrix
    feature_norm_squared_vec : ??
                               ??

    return : array (d*n, 1)
             thresholded weights as a vector
    """

    X_out = feature_block_masked@W_in

    W_in_row = np.reshape(W_in, (d,n), order='F')
    W_in_coeff = np.sum(X*W_in_row, axis=0) / feature_norm_squared_vec

    coeffs = np.zeros((n, 1))

    # No loop
    # coeff_delta = np.reshape(np.array(W_in_coeff[X_idx] - (y[X_idx] / feature_norm_squared_vec[X_idx])), (-1, 1))
    # y_tmp = self.wthresh(coeff_delta, gamma_vec)
    # a = np.array(y[X_idx] / feature_norm_squared_vec[X_idx])
    # print('a: {}'.format(a.shape))
    # coeffs[X_idx] = np.array(y[X_idx] / feature_norm_squared_vec[X_idx]) + y_tmp

    # Same with loop
    for id in X_idx:
        tmp = W_in_coeff[id] - (y[id]/feature_norm_squared_vec[id])
        y_temp = wthresh(tmp, gamma_vec[:,id])
        coeffs[id] = (y[id]/feature_norm_squared_vec[id])+y_temp # checked

    tmp = X@np.diagflat(coeffs) + np.reshape(X_out, (d, n), order='F') #checked

    W_out = np.reshape(W_in, (d, n), order='F')
    W_out[:, X_idx] = tmp[:, X_idx] # checked
    W_out = np.reshape(W_out, (d*n, 1), order='F') # checked
    return W_out

def block_clipping(W_in, d, n, lambda_):
    """
    eq. 40

    W_in : array (n*d, 1)
           weights in
    d : int
        feature dimensions
    n : int
        number of nodes
    lambda_ : ??
              clipping threshold

    return : array (n*d, 1)
             clipped weights
    """

    tmp = np.reshape(W_in, (d, n), order='F') # checked
    X_norm = np.linalg.norm(tmp, axis=0)

    factor = np.ones((1, n))
    idx_exceed = np.where(X_norm > lambda_)[0]
    if len(idx_exceed) > 0:
        np.put(factor, idx_exceed, (lambda_/X_norm[idx_exceed]))

    W_out = np.reshape((tmp @ np.diagflat(factor)), (n*d, 1), order='F')

    return W_out


def fit_primal_dual(X, y, R, num_iter, clipping_threshold, iteration=0, W_average=None, ):
    """
    R = A
    """

    n, d = X.shape
    R = np.triu(R, 1)
    nro_edges = len(R[R>0])

    running_average = np.zeros((n*d, 1))

    np.set_printoptions(edgeitems=5, precision=4)
    # D (eq 11.)
    G = nx.from_numpy_matrix(R, create_using=nx.DiGraph())
    D = nx.incidence_matrix(G, oriented=True).transpose()

    W_edge = np.zeros((nro_edges,1))

    # TODO pick indicies corresponding to existing labels
    drop_idx = [22,17,21,8,6,4]
    #X_sampling_idx = np.delete(np.arange(n), drop_idx)
    X_sampling_idx = np.arange(n)
    X_sampling = X.T
    y_sampling = y

    # Mask and normalised feature vectors
    mask = kron(eye(n,n), np.ones((d,d)))
    feature_norm_squared_vec = np.sum(X_sampling**2, axis=0) # checked
    feature_vec = np.reshape(X_sampling, (n*d, 1), order='F');

    feature_block_masked =  np.eye(n*d, n*d) - ( ((feature_vec@feature_vec.transpose()) * mask.toarray()) @ spsolve(csc_matrix(kron(np.diagflat(feature_norm_squared_vec), eye(d,d, format='csc'))), eye(n*d, n*d, format='csc'))) # checked, dense

    D_block = kron(D.transpose(), eye(d, d)).transpose() #sparse

    # TODO better to use random, replace 0 with small value or skip the batch?
    abs_sum_col = np.absolute(D).sum(axis=1)
    # eq. 14 (why differs?)
    # TOCO rename to match with the article
    # lambda_block = sigma
    lambda_block = kron(np.diagflat(np.nan_to_num(1./abs_sum_col)), eye(d, d)) # sparse

    abs_sum_row = np.absolute(D).sum(axis=0).ravel()

    # gamma = T
    gamma_vec=np.nan_to_num(1./abs_sum_row)
    gamma_block = kron(np.diagflat(gamma_vec, 0), eye(d, d))

    w_hat = np.zeros((n*d, 1))
    u_hat = np.zeros((nro_edges*d, 1))

    for i in np.arange(num_iter):
        # alg. 2
        w_new = w_hat - .5 * gamma_block @ (D_block.transpose() @ u_hat)
        # alg. 3, eq. 19
        w_new = block_thresholding(w_new,
                                   X_sampling_idx,
                                   y_sampling,
                                   X_sampling,
                                   d,
                                   n,
                                   gamma_vec,
                                   feature_block_masked,
                                   feature_norm_squared_vec)

        w_tilde = 2 * w_new - w_hat
        w_hat = w_new

        # alg. 4
        u_new = u_hat + .5 * lambda_block @ (D_block @ w_tilde)
        # alg. 5
        u_hat = block_clipping(u_new, d, nro_edges, clipping_threshold)

        it = iteration + i
        running_average = (running_average * it + w_hat) / (it+1)

    W_average = get_w_average(running_average, W_average, d, n, it)

    return W_average, it

def get_w_average(running_average, W_average, d, n, it):
    """ Get running average of weights """
    # Traditional preidction for unseen data
    W = np.mean(np.reshape(running_average, (d,n), order='F').T, axis=0)
    if W_average is None:
        W_average = W
    else:
        W_average = (W_average * it + W) / (it+1)
    #self.W = np.mean(np.reshape(w_hat, (d,n), order='F').T, axis=0)

    return W_average


def wthresh(a, thresh):
    """ Soft threshold """
    a = np.reshape(a, (-1, 1))
    res = np.array(np.abs(a) - thresh)
    res[res<0] = 0
    sign = np.array(np.sign(a))

    return sign * res
