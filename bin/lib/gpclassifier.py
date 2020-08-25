import numpy as np
import logging
from sklearn.base import BaseEstimator

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct, PairwiseKernel

#import gpflow
from sklearn.exceptions import NotFittedError

class GPClassifier(BaseEstimator):

    fitted = False
    limit = .5
    model = None

    def __init__(self, params=None,model=None,limit=.5,noise_level=5):
        """ Init """

        if model is None:
            kernel = PairwiseKernel(metric='laplacian') *  DotProduct() + WhiteKernel(noise_level=noise_level)
            self.model = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
        else:
            self.fitted = True
            self.model = model

        if limit is not None:
            self.limit = limit

    def save(self, model_filename):
        """ Save """
        pass

    def get_model(self):
        return self.model

    def fit(self, X, y, val_X=None, val_y=None):
        """ Fit """
        # self.model = gpflow.models.VGP((X, y),
        #                                kernel=gpflow.kernels.SquaredExponential(),
        #                                likelihood=gpflow.likelihoods.Bernoulli())
        # self.opt = gpflow.optimizers.Scipy()
        #
        # self.opt.minimize(self.model.training_loss,
        #                   variables=self.model.trainable_variables,
        #                   options=dict(maxiter=25),
        #                   method="L-BFGS-B")
        self.model.fit(X, y)
        self.fitted = True

        return self

    def predict(self, X, type='int', **params):
        """ Predict """
        if not self.fitted:
            raise NotFittedError()

        logging.info('Predicting fit classifier using limit {}...'.format(self.limit))
        self.y_pred_proba = self.model.predict_proba(X)

        if type == 'bool':
            return np.fromiter(map(lambda x: False if x < self.limit else True, self.y_pred_proba[:,1]), dtype=np.bool)

        # Scale to [0 1]
        return np.fromiter(map(lambda x: 0 if x < self.limit else 1, self.y_pred_proba[:,1]), dtype=np.int)


    def predict_proba(self, X):
        """ Predict proba """
        if not self.fitted:
            raise NotFittedError()

        self.y_pred_proba = self.model.predict_proba(X)

        return self.y_pred_proba
