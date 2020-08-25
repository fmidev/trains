# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import logging
from sklearn.exceptions import NotFittedError

class Selector:
    """ Data selector based on binary classifier """
    full = False
    lstm = False
    regressor = None
    classifier = None
    limit = None

    def __init__(self, classifier, regressor, y=None):
        self.classifier = classifier
        self.regressor = regressor
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
        self.y_pred_proba = self.predict_proba(X)
        #prediction = self.classifier.predict(X)

        if 'limit' in params: limit = params['limit']
        elif self.limit is not None: limit = self.limit
        else: limit = .5

        logging.info('Predicting by selector classifier using limit {}...'.format(limit))

        type = params.get('type', 'bool')
        if type == 'bool':
            self.y_pred = np.fromiter(map(lambda x: False if x < limit else True, self.y_pred_proba[:,1]), dtype=np.bool)
        else:
            self.y_pred = np.fromiter(map(lambda x: 0 if x < limit else 1, self.y_pred_proba[:,1]), dtype=np.int32)

        return self.y_pred

    def predict_proba(self, X):
        """
        Predict with probabilities

        X : np.array
        """
        logging.info('Predicting with probabilities by selector classifier...')
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

        self.full = True
        if not self.full:

            y_pred = self.predict(X)

            # # LSTM is omitting first time steps
            if self.lstm:
                # TODO not tested
                Xt = X[(len(X)-len(y_pred)):]
                ret = Xt[y_pred]
                delay_data = self.delay_data[(len(X)-len(y_pred)):len(X)]
                dt = delay_data[y_pred].reshape((-1,1))
            else:
                self.regressor.set_y(self.delay_data[:len(X)][y_pred])
                ret = X[y_pred]
        else:
            ret = X

        return ret

    def set_y(self, data):
        """ Set delay data """
        self.delay_data = data
        self.regressor.set_y(data)

    def set_params(self, **params):
        """ Set params """
        self.__dict__.update(params)

    def set_regressor(self, regressor):
        """ Set regressor """
        self.regressor = regressor


class Regressor:
    """ Regressor wrapper class """

    classifier = None
    model = None
    y = None
    fitted = False

    def __init__(self, model, classifier=None, fitted=False):
        """ Init """
        self.model = model
        self.classifier = classifier
        self.fitted = fitted

    def fit(self, X, y=None):
        """
        Fit model

        NOTE: y is taken from the class variable and given argument is ignored

        """
        logging.info('Fitting regressor...')

        # TODO if fitted with pipe, the line should be in use
        #y = self.y

        if not self.fitted:
            self.model.fit(X, y)

    def predict(self, X, X_class, **params):
        """
        Predict using first classifier and then regression

        X        : array
                   regression features
        X_class  : array
                   classification features
        **params : any
                   passed to classifier predict
        """
        logging.info('Predicting by regressor...')

        if self.classifier is None:
            raise Exception("Classifier not set. Call set_classifier(classifier) with fitted classifier")

        pred_binary = self.classifier.predict(X_class, **params)
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

    def set_y(self, y):
        """ Set Y """
        self.y = y
