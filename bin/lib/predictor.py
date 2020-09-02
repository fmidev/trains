import sys, os, argparse, logging, json

import itertools
from collections import OrderedDict

import math
import numpy as np
import pandas as pd

# import tensorflow as tf
# from tensorflow.python.estimator.export import export
# from tensorflow.python.framework import constant_op
#from tensorflow.contrib import predictor

#from keras.preprocessing.sequence import TimeseriesGenerator

class PredictionError(Exception):
   """Prediction error exception"""
   pass

class Predictor():

    y_pred_bin = None
    y_pred_bin_proba = None
    mean_delay = 3.375953418071136

    def __init__(self, io, model_loader, options, STATION_SPECIFIC_CLASSIFIER=True, STATION_SPECIFIC_REGRESSOR=False):
        self.io = io
        self.model_loader = model_loader
        self.options = options
        self.station_specific_classifier = STATION_SPECIFIC_CLASSIFIER
        self.station_specific_regressor = STATION_SPECIFIC_REGRESSOR
        self.regressor_save_file = options.save_path+'/regressor.pkl'
        self.classifier_save_file = options.save_path+'/classifier.pkl'

    def _normalise_data(self, scaler, data):
        """
        Normalise features with given scaler

        scaler : StandardScaler
                 scaler with scikit API
        data : DataFrame
               data to be scaled

        return DataFrame with scaled features
        """
        try:
            data.reset_index(inplace=True)
        except ValueError:
            data.reset_index(inplace=True, drop=True)

        non_scaled_data = data.loc[:,self.options.meta_params + self.options.label_params]
        scaled_features = pd.DataFrame(scaler.transform(data.loc[:,self.options.feature_params]),
                                       columns=self.options.feature_params)
        data = pd.concat([non_scaled_data, scaled_features], axis=1)
        return data

    def pred_keras(self, data):
        """
        Run Keras prediction

        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation )

        return lst target, lst prediction
        """
        model = self.model_loader.load_keras_model(self.options.save_path, self.options.save_file)

        if self.options.normalize:
            xscaler, yscaler, _, _, _ = self.model_loader.load_scalers(self.options.save_path)
            data = self._normalise_data(xscaler, data)

        y_pred, target = [], []
        logging.info('Predicting using keras... ')

        batch_size = 512

        # print(data.loc[:, self.options.feature_params].dropna())
        try:
            data_gen = TimeseriesGenerator(data.loc[:, self.options.feature_params].values,
                                           np.ones(len(data.loc[:,self.options.feature_params].values)),
                                           length=self.options.time_steps,
                                           sampling_rate=1,
                                           batch_size=batch_size)
            y_pred = model.predict_generator(data_gen)
        except ValueError as e:
            raise PredictionError(e)

        if self.options.normalize:
            y_pred = yscaler.inverse_transform(y_pred)

        target = data.loc[self.options.time_steps:, self.options.label_params].values

        return y_pred, None, None

    def pred_tf(self, times, data):
        """
        Run TensorFlow prediction

        times : lst
                list of time steps to which prediction is to be done
        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation )

        return lst target, lst prediction
        """
        sess, op_y_pred, X = self.model_loader.load_tf_model(self.options.save_path, self.options.save_file)

        if self.options.normalize:
            xscaler, yscaler, _, _, _ = self.model_loader.load_scalers(self.options.save_path)
            data = self._normalise_data(xscaler, data)

        y_pred, target = [], []
        logging.info('Predicting using tf... ')
        start = 0
        end = self.options.time_steps
        first = True
        while end <= len(times):
            input_batch, target_batch = self.io.extract_batch(data,
                                                              self.options.time_steps,
                                                              batch_size=-1,
                                                              pad_strategy=self.options.pad_strategy,
                                                              quantile=self.options.quantile,
                                                              label_params=self.options.label_params,
                                                              feature_params=self.options.feature_params,
                                                              start=start,
                                                              end=end)
            if len(input_batch) < self.options.time_steps:
                break

            feed_dict={X: input_batch}
            y_pred_batch = sess.run(op_y_pred, feed_dict=feed_dict).ravel()
            if self.options.normalize:
                y_pred_batch = yscaler.inverse_transform(y_pred_batch)

            if first:
                y_pred = list(y_pred_batch)
                target = list(target_batch.ravel())
                first = False
            else:
                y_pred.append(y_pred_batch[-1])
                target.append(target_batch[-1])

            start += 1
            end += 1

            if end%100 == 0:
                logging.info('...step {}/{}'.format(end, len(times)))

        return y_pred, None, None

    def pred_scikit(self, data):
        """
        Run SciKit prediction

        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation )

        return lst target, lst prediction
        """
        predictor = self.model_loader.load_scikit_model(self.options.save_path+'/model.pkl')

        if self.options.normalize:
            xscaler, yscaler, _, _, _ = self.model_loader.load_scalers(self.options.save_path)
            data = self._normalise_data(xscaler, data)

        # Pick feature and label data from all data
        target = data.loc[:,self.options.label_params].astype(np.float64).values.ravel()
        features = data.loc[:, self.options.feature_params].astype(np.float64).values

        y_pred = predictor.predict(features)

        if self.options.normalize:
            y_pred = yscaler.inverse_transform(y_pred)

        return y_pred, None, None

    def pred_llasso(self, data):
        """
        Run LocalizedLasso prediction

        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation )

        return lst target, lst prediction
        """
        predictor = self.model_loader.load_scikit_model(self.options.save_path+'/'+self.options.save_file)

        if self.options.normalize:
            xscaler, yscaler, _, _, _ = self.model_loader.load_scalers(self.options.save_path)
            data = self._normalise_data(xscaler, data)

        # Pick feature and label data from all data
        target = data.loc[:,self.options.label_params].astype(np.float64).values.ravel()
        features = data.loc[:, self.options.feature_params].astype(np.float64).values

        y_pred, weights = predictor.predict(features, data.loc[:, 'trainstation'].values)

        if self.options.normalize:
            y_pred = yscaler.inverse_transform(y_pred)

        self.weights = weights

        return y_pred, None, None

    def pred_dual(self, data):
        """
        Run dual prediction. First run classification and then regression prediction.

        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation )

        return lst target, lst prediction
        """

        classifier = self.model_loader.load_scikit_model(self.classifier_save_file, True, 'classifier')

        classifier.limit = self.options.class_limit
        #classifier.set_y(data.loc[:, self.options.label_params])
        #regressor = classifier.regressor

        regressor = self.model_loader.load_scikit_model(self.regressor_save_file, True, 'regressor')

        # Pick feature and label data from all data
        features_classifier = data.loc[:, self.options.classifier_feature_params].copy().astype(np.float64).values
        features_regressor = data.loc[:, self.options.regressor_feature_params].copy().astype(np.float64).values

        if self.options.normalize_classifier or self.options.normalize_regressor:
            _, _, xscaler_classifier, xscaler_regressor, yscaler_regressor = self.model_loader.load_scalers(self.options.save_path)
            if self.options.normalize_classifier:
                if xscaler_classifier is not None:
                    logging.info('Normalizing classification data...')
                    features_classifier = xscaler_classifier.transform(features_classifier)

            if self.options.normalize_regressor:
                if xscaler_regressor is not None:
                    logging.info('Normalizing regressor data...')
                    features_regressor = xscaler_regressor.transform(features_regressor)
                    #data = self._normalise_data(xscaler, data)

        self.y_pred_bin = classifier.predict(features_classifier, type='int')
        self.y_pred_bin_proba = classifier.y_pred_proba
        #X = features[(len(features)-len(self.y_pred_bin)):]

        logging.info('Predicting with regressor...')

        y_pred_reg = regressor.predict(features_regressor).astype(np.float64)

        if self.options.normalize_regressor and yscaler_regressor is not None:
            logging.info('Inverse transform for y_pred_reg')
            y_pred_reg = yscaler_regressor.inverse_transform(y_pred_reg)

        #a = np.fromiter(map(lambda x: 0 if not x else 1, y_pred_bin), dtype=np.int32)

        # LSTM do not have first time_steps
        #X = X[(len(X)-len(self.y_pred_bin)):]

        # TODO do we want this step?
        # Pick only severe values
        #y_pred = np.choose(self.y_pred_bin, (np.full(X.shape[0], self.mean_delay), y_pred_reg))
        # Use all values
        y_pred = y_pred_reg

        #target = data.loc[(len(features)-len(self.y_pred_bin)):, self.options.label_params].values.ravel()

        return y_pred, self.y_pred_bin, self.y_pred_bin_proba



    def pred(self, times=None, data=None):
        """
        Run SciKit prediction

        times : lst
                list of time steps to which prediction is to be done
        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation)

        return lst target, lst prediction
        """
        if self.options.model_type == 'keras':
            return self.pred_keras(data)
        elif self.options.model_type == 'tf':
            return self.pred_tf(times, data)
        elif self.options.model_type in ['llasso', 'nlasso']:
            return self.pred_llasso(data)
        elif self.options.model_type == 'dual':
            return self.pred_dual(data)

        return self.pred_scikit(data)
