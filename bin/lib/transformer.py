# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import logging
from sklearn.exceptions import NotFittedError

class Selector:
    """ Data selector based on binary classifier """
    full = False

    def __init__(self, classifier, y=None):
        self.classifier = classifier
        self.delay_data = y

    def fit(self, X, y):
        """ Fit the selector if not fitted already """
        logging.info('Fitting selector classifier...')
        # If prediction not raise an exception, the classifier is fitted
        try:
            self.classifier.predict(X[0:1,:])
        except NotFittedError as e:
            logging.info('...not fitted, fitting...')
            self.classifier.fit(X, y)

    def predict(self, X, **params):
        """
        Predict

        X : np.array
        type : str
               if 'bool', return boolean array, else return -1/1 values
        """
        logging.info('Predicting by selector classifier...')

        prediction = self.classifier.predict(X)
        type = params.get('type', 'bool')
        if type == 'bool':
            return np.fromiter(map(lambda x: False if x < 0 else True, prediction), dtype=np.bool)
        else:
            return prediction

    def predict_proba(self, X):
        """
        Predict with probabilities

        X : np.array
        """
        logging.info('Predicting with probabilities by selector classifier...')
        prediction = self.classifier.predict(X)

        return self.classifier.predict_proba(X)


    def fit_predict(self, X, y):
        """ Fit if not fitted and predict """
        logging.info('Fitting and predicting by selector classifier...')
        self.classifier.fit_predict(X, y)

    def fit_transform(self, X, y):
        """ Fit if not fitted and transform """
        logging.info('Fitting and transforming by selector...')
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X, **params):
        """ Transform X """
        logging.info('Transforming by selector...')

        if not self.full:
            y_pred = self.predict(X)

            # LSTM is omitting first time steps
            Xt = X[(len(X)-len(y_pred)):]
            Xt = Xt[y_pred]
            delay_data = self.delay_data[(len(X)-len(y_pred)):]
            dt = delay_data[y_pred].reshape((-1,1))

            a = np.fromiter(map(lambda x: 0 if not x else 1, y_pred), dtype=np.int32)
            
            ret = np.concatenate((dt, Xt), axis=1)

            return np.choose(a, (np.zeros(X.shape[0]), self.model.predict(X)))
        else:
            ret = X

        return ret

    def set_y(self, data):
        """ Set delay data """
        self.delay_data = data

    def set_params(self, **params):
        """ Set params """
        self.__dict__.update(params)


class Regressor:
    """ Regressor wrapper class """

    classifier = None
    model = None

    def __init__(self, model):
        """ Init """
        self.model = model

    def fit(self, X, y=None):
        """
        Fit model

        NOTE: y is taken from the first column of X and corresponding argument is ignored.
        X is taken from X[:,1:]
        """
        logging.info('Fitting regressor...')

        X = X[:,1:]
        y = X[:,0]

        self.model.fit(X, y)

    def predict(self, X, **params):
        """
        Predict using first classifier and then regression
        """
        logging.info('Predicting by regressor...')

        if self.classifier is None:
            raise Exception("Classifier not set. Call set_classifier(classifier) with fitted classifier")

        pred_binary = self.classifier.predict(X, **params)
        a = np.fromiter(map(lambda x: 0 if not x else 1, pred_binary), dtype=np.int32)
        # LSTM do not have first time_steps
        X = X[(len(X)-len(a)):]
        return np.choose(a, (np.zeros(X.shape[0]), self.model.predict(X)))

    def fit_predict(self, X, y=None):
        """ Fit and predict """
        logging.info('Fitting and predicting by regressor...')
        self.fit(X, y)
        return self.predict()

    def get_feature_importances(self):
        """ Get feature importances from RFR """
        return self.model.feature_importances_

    def set_classifier(self, classifier):
        """
        Set fitted classifier to predict whether delays are going to happen at all
        """
        self.classifier = classifier
