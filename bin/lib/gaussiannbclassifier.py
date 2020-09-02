import numpy as np
import logging
from sklearn.base import BaseEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.exceptions import NotFittedError

class GaussianNBClassifier(BaseEstimator):

    fitted = False
    limit = .5

    def __init__(self, limit=None, model=None):
        """ Init """

        if limit is not None:
            self.limit = limit

        if model is None:
            self.model = GaussianNB()
        else:
            self.fitted = True
            self.model = model

    def save(self, model_filename):
        """ Save """
        pass

    def get_model(self):
        return self.model

    def fit(self, X, y, val_X=None, val_y=None):
        """ Fit """
        self.model.fit(X, y)
        self.fitted = True
        return None

    def partial_fit(self, X, y, val_X=None, val_y=None):
        """ Fit """
        self.model.partial_fit(X, y)
        self.fitted = True
        return None

    def predict(self, X, type='int'):
        """ Predict """
        if not self.fitted:
            raise NotFittedError()

        y_pred = self.model.predict(X)
        self.y_pred_proba = self.predict_proba(X)

        if type == 'bool':
            return np.fromiter(map(lambda x: False if x < self.limit else True, y_pred), dtype=np.bool)

        return y_pred

        #y_pred_proba = self.model.predict_proba(X)
        #self.y_pred_proba = y_pred_proba


        #if type == 'bool':
        #    return np.fromiter(map(lambda x: False if x < self.limit else True, y_pred_proba[:,1]), dtype=np.bool)

        # Scale to [0 1]
        #return np.fromiter(map(lambda x: 0 if x < self.limit else 1, y_pred_proba[:,1]), dtype=np.int)


    def predict_proba(self, X):
        """ Predict proba """
        if not self.fitted:
            raise NotFittedError()

        y_pred_proba = self.model.predict_proba(X)

        return y_pred_proba
