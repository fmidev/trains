import sys, os, logging
import gpflow
from sklearn.base import BaseEstimator
import numpy as np

class GP(BaseEstimator):
    """
    Scikit API wrapper for gpflow model
    """
    estimator_params = ("Z", "kern", "minibatch_size", "whiten", "length_scale", "likelihood")
    def __init__(self, dim, length_scale=1, kern=None, likelihood=None, Z=None, minibatch_size=100, whiten=True):
        """
        Initialization
        """
        self.dim = dim
        self.length_scale = length_scale
        self.kern = kern
        if kern is None:
            self.kern = gpflow.kernels.Matern52(dim, lengthscales=self.length_scale)
        self.likelihood = likelihood
        if likelihood is None:
            self.likelihood = gpflow.likelihoods.Gaussian()
        self.Z = Z
        if Z is None:
            self.Z = np.random.rand(5, self.dim)
        self.minibatch_size = minibatch_size
        self.whiten = whiten

    def fit(self, X, y):

        self.model = gpflow.models.SVGP(X,
                                        y,
                                        kern=self.kern,
                                        likelihood=self.likelihood,
                                        Z=self.Z,
                                        #Z=X_train.copy(),
                                        minibatch_size=self.minibatch_size,
                                        whiten=self.whiten
                                        )
                                        #model.likelihood.variance = 0.01

        self.model.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.model)

    def predict_f(self, X):
        return self.predict(X)

    def predict(self, X):
        y_pred, var = self.model.predict_f(X)
        return y_pred, var

    def save(self, fname):
        saver = gpflow.saver.Saver()
        if os.path.exists(fname):
            os.remove(fname)
        saver.save(fname, self.model)

    def load(self, fname):
        pass


    # def get_params(self,):
    #     pass
    #
    # def set_params(self,):
    #     pass
