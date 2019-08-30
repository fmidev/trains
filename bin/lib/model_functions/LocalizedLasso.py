import numpy as np
import scipy.sparse as sp
from numpy.matlib import repmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator


class LocalizedLasso(BaseEstimator):
    """ Localized Lasso with iterative-least squares optimization

    Skeleton of the code taken from: https://riken-yamada.github.io/localizedlasso.html
    Paper: http://proceedings.mlr.press/v54/yamada17a/yamada17a.pdf
    """
    def __init__(self, num_iter=10, lambda_net=1, lambda_exc=0.01, biasflag=0, R=None):
        """
            num_iter: int
               The number of iteration for iterative-least squares update: (default: 10)
            lambda_net: double
               The regularization parameter for the network regularization term (default:1)
            lambda_exc: double
               The regularization parameter for the exclusive regularization (l12) term (default:0.01)
            biasflag : int
               1: Add bias term b_i
               0: No bias term
            R : n x n dimensional matrix
               connection between rows

            W: (d + 1) x n dimensional matrix
               i-th column vector corresponds to the model for w_i
            vecW: double
               dn + 1 dimensional parameter vector vec(W)
            Yest: double
               Estimated training output vector.
            """

        self.num_iter   = num_iter
        self.lambda_net = lambda_net
        self.lambda_exc = lambda_exc
        self.biasflag = biasflag
        self.R = R

    def fetch_connections(self, data):
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

        self.R = np.array(R)
        self.station_index = station_index

        return self.R

    def pick_W(self, stations):
        """
        Pick weights from training set based on train stations. Note that
        weights may occur at several rows since R matrix defines how
        rows are connected to stations.

        stations : lst (n,)
                   list of train station of each row in X

        return ndarray d x n
        """
        W = []
        for row_station in stations:
            try:
                W.append(self.W[:, self.station_index.index(row_station)])
            except ValueError:
                #probably not correct way to take the mean
                W.append(self.W.mean(1))

        return np.array(W).T

    def fit(self, X, y, stations, mode='regression'):
        """
        Fit the model

        X : ndarray [n x d]
            features
        y : ndarray or list [n x 1]
            lables
        stations : lst [n,]
                   list of train station of each row in X and y
        """

        # Make connections how rows are connected to stations
        R = self.make_r(stations)

        if mode == 'regression':
            self.fit_regression(X.T, y.T, R)


    def predict(self, X, stations):
        """
        Predict

        X : ndarray [n x d]
            features
        stations : lst [n,]
                   list of train station of each row in X and y

        return y_pred (lst [n,1]), weights (ndarray [n,d])
        """

        # Make connections how rows are connected to stations
        R = self.make_r(stations)

        # Pick weights based on stations of each row in X
        W = self.pick_W(stations)

        return self.prediction(X, R, W)

        # Paper did this this way:
        #n, d = X.shape
        #for i in range(0, n):
        #    self.prediction(X[i,:], self.R[:,i])

    #Prediction with Weber optimization
    def prediction(self, X, R, W):
        """
        Prediction using Weber optmisasion

        X: ndarray [n, d]
           features
        R : ndarray [n, n]
            connection matrix telling each row's relation to each others
        W : ndarray [d, n]
            weights

        return y_pred (lst [n,1]), weights (ndarray [n,d])
        """
        [d,n] = W.shape
        # print('Prediction')
        # print(self.W.shape)
        # print(W.shape)

        wte = np.zeros((1,d))
        # print('wte shape: {}'.format(wte.shape))
        loss_weber = np.zeros((20,1))

        if np.sum(R) == 0:
            wte = self.W.mean(1)[np.newaxis,:]
        else:
            for k in range(0,20):
                dist2 = cdist(wte, W.transpose())
                invdist2 = R/(2*dist2 + 10e-5)

                sum_dist2 = np.sum(invdist2)
                wte = np.dot(invdist2, W.transpose())/sum_dist2
                # if(k == 0):
                #     print('dist2 shape: {}'.format(dist2.shape))
                #     print('R shape: {}'.format(R.shape))
                #     print('invdist2 shape: {}'.format(invdist2.shape))
                #     print('wte shape: {}'.format(wte.shape))
                loss_weber[k] = np.sum(R*dist2)

        if self.biasflag == 1:
            yte = (wte[0][0:(d - 1)] * X).sum() + wte[0][d-1]
        else:
            # print(wte[0].shape)
            # print(X.shape)
            # print((wte*X).shape)
            #yte = (wte[0]*X).sum()
            yte = (wte*X).sum(1)

        return yte, wte

    #Regression
    def fit_regression(self,X,Y,R):
        """
        Fit regression case

        X : ndarray [n, d]
            features
        Y : list [n, 1]
            labels
        R : ndarray [n, n]
            connection matrix telling each row's relation to each others
        """
        [d,ntr0] = X.shape

        if self.biasflag == 1:
            Xtr = np.concatenate((X, np.ones((1,ntr0))), axis=0)
        else:
            Xtr = X

        Ytr = Y

        [d,ntr] = Xtr.shape
        dntr = d*ntr

        vecW = np.ones((dntr,1))
        index = np.arange(0,dntr)
        val = np.ones(dntr)
        # D = lambda_exc * Fg + lambda_net * Fe
        D = sp.csc_matrix((val,(index,index)),shape=(dntr,dntr))

        #Generate input matrix
        A = np.zeros((ntr,dntr)) #sp.csc_matrix((ntr,dntr))

        for ii in range(0,ntr):
            ind = range(ii,dntr,ntr)
            A[ii,ind] = Xtr[:,ii]

        A = sp.csc_matrix(A)

        one_ntr = np.ones(ntr)
        I_ntr = sp.diags(one_ntr, 0, format='csc')

        one_d = np.ones(d)
        I_d = sp.diags(one_d, 0, format='csc')

        fval = np.zeros(self.num_iter)
        for iter in range(0,self.num_iter):

            DinvA = spsolve(D, A.transpose())
            B = I_ntr + A.dot(DinvA)
            tmp = spsolve(B,Ytr)
            vecW = DinvA.dot(tmp)

            W = np.reshape(vecW,(ntr,d),order='F')

            tmpNet = cdist(W,W)
            tmp = tmpNet*R

            #Network regularization
            U_net = tmp.sum()

            tmp = 0.5 / (tmpNet + 10e-10) * R

            td1 = sp.diags(tmp.sum(0),0)
            td2 = sp.diags(tmp.sum(1),0)

            AA = td1 + td2 - 2.0*tmp
            AA = (AA + AA.transpose())*0.5 + 0.001*sp.eye(ntr,ntr)

            # Update Fe
            Fe = kron(I_d, AA, format='csc')

            #Exclusive regularization
            if self.biasflag == 1:
                tmp = abs(W[:,0:(d-1)]).sum(1)
                U_exc = (tmp*tmp).sum()

                tmp_repmat = repmat(np.c_[tmp],1,d)
                tmp_repmat[:,d-1] = 0
                tmp = tmp_repmat.flatten(order='F')
            else:
                tmp = abs(W).sum(1)
                U_exc = (tmp*tmp).sum()

                tmp_repmat = repmat(np.c_[tmp],1,d)
                tmp = tmp_repmat.flatten(order='F')

            self.coef_ = abs(W).sum(0)

            # Update Fg
            tmp = tmp / (abs(vecW) + 10e-10)
            Fg = sp.diags(tmp,0,format='csc')

            # lambda_exc * Fg + lambda_net * Fe
            D = self.lambda_net*Fe + self.lambda_exc*Fg
            fval[iter] = ((Ytr -A.dot(vecW))**2).sum() + self.lambda_net*U_net + self.lambda_exc*U_exc

            print('fval: {}'.format(fval[iter]))

        self.vecW = vecW
        self.W = np.reshape(vecW,(ntr,d),order='F').transpose()
        self.Yest = A.dot(vecW)

    # def fit_clustering(self,X,R):
    #
    #     [d,ntr] = X.shape
    #     dntr = d*ntr
    #
    #     vecW = np.ones((dntr,1))
    #     index = np.arange(0,dntr)
    #     val = np.ones(dntr)
    #     D = sp.csc_matrix((val,(index,index)),shape=(dntr,dntr))
    #
    #     #Generate input matrix
    #     vecXtr = X.transpose().flatten(order='F')
    #
    #     one_d = np.ones(d)
    #     I_d = sp.diags(one_d,0,format='csc')
    #
    #     one_dntr = np.ones(dntr)
    #     I_dntr = sp.diags(one_dntr,0,format='csc')
    #
    #     fval = np.zeros(self.num_iter)
    #     for iter in range(0,self.num_iter):
    #
    #         vecW = spsolve(D,vecXtr)
    #
    #         W = np.reshape(vecW,(ntr,d),order='F')
    #
    #         tmpNet = cdist(W,W)
    #         tmp = tmpNet*R
    #
    #         #Network regularization
    #         U_net = tmp.sum()
    #
    #         tmp = 0.5 / (tmpNet + 10e-10) * R
    #
    #         td1 = sp.diags(tmp.sum(0),0)
    #         td2 = sp.diags(tmp.sum(1),0)
    #
    #         AA = td1 + td2 - 2.0*tmp
    #         AA = (AA + AA.transpose())*0.5 + 0.00001*sp.eye(ntr,ntr)
    #
    #         Fe = kron(I_d,AA,format='csc')
    #
    #         #Exclusive regularization
    #         tmp = abs(W).sum(1)
    #         U_exc = (tmp*tmp).sum()
    #
    #         tmp_repmat = repmat(np.c_[tmp],1,d)
    #         tmp = tmp_repmat.flatten(order='F')
    #         tmp = tmp / (abs(vecW) + 10e-10)
    #         Fg = sp.diags(tmp,0,format='csc')
    #
    #         D = I_dntr + self.lambda_net*Fe + self.lambda_exc*Fg
    #
    #         fval[iter] = ((vecXtr - vecW)**2).sum() + self.lambda_net*U_net + self.lambda_exc*U_exc
    #
    #         print('fval: {}'.format(fval[iter]))
    #
    #     self.vecW = vecW
    #     self.W = np.reshape(vecW,(ntr,d),order='F').transpose()
