import numpy as np
import logging
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError

class SVCClassifier(BaseEstimator):

    fitted = False
    limit = .1

    def __init__(self, params, limit=None, model=None):
        """ Init """

        if model is None:
            self.model = SVC(**params)
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
        self.model.fit(X, y)
        self.fitted = True

        return None

    def predict(self, X, type='int'):
        """ Predict """
        if not self.fitted:
            raise NotFittedError()

        y_pred_proba = self.model.predict_proba(X)
        self.y_pred_proba = y_pred_proba
        print(self.limit)
        if type == 'bool':
            return np.fromiter(map(lambda x: False if x < self.limit else True, y_pred_proba[:,1]), dtype=np.bool)

        # Scale to [0 1]
        return np.fromiter(map(lambda x: 0 if x < self.limit else 1, y_pred_proba[:,1]), dtype=np.int)


    def predict_proba(self, X):
        """ Predict proba """
        if not self.fitted:
            raise NotFittedError()

        y_pred_proba = self.model.predict_proba(X)

        return y_pred_proba
