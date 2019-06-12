import sys, os, argparse, logging, json

import itertools
from collections import OrderedDict

import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.estimator.export import export
from tensorflow.python.framework import constant_op
from tensorflow.contrib import predictor

from keras.preprocessing.sequence import TimeseriesGenerator

class Predictor():

    def __init__(self, io, model_loader, options):
        self.io = io
        self.model_loader = model_loader
        self.options = options

    def _normalise_data(self, scaler, data):
        """
        Normalise features with given scaler

        scaler : StandardScaler
                 scaler with scikit API
        data : DataFrame
               data to be scaled

        return DataFrame with scaled features
        """
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
            xscaler, yscaler = self.model_loader.load_scalers(self.options.save_path)
            data = self._normalise_data(xscaler, data)

        y_pred, target = [], []
        logging.info('Predicting using keras... ')

        batch_size = 512

        # print(data.loc[:, self.options.feature_params].dropna())
        data_gen = TimeseriesGenerator(data.loc[:, self.options.feature_params].values,
        #                               data.loc[:, self.options.label_params].values,
                                       np.ones(len(data.loc[:, self.options.feature_params].values)),
                                       length=self.options.time_steps,
                                       sampling_rate=1,
                                       batch_size=batch_size)
        y_pred = model.predict_generator(data_gen)

        if self.options.normalize:
            y_pred = yscaler.inverse_transform(y_pred)

        target = data.loc[self.options.time_steps:, self.options.label_params].values

        print(target.shape)

        return target, y_pred

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
            xscaler, yscaler = self.model_loader.load_scalers(self.options.save_path)
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

        return target, y_pred

    def pred_scikit(self, times, data):
        """
        Run SciKit prediction

        times : lst
                list of time steps to which prediction is to be done
        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation )

        return lst target, lst prediction
        """
        predictor = self.model_loader.load_scikit_model(self.options.save_path, self.options.save_file)

        if self.options.normalize:
            xscaler, yscaler = self.model_loader.load_scalers(self.options.save_path)
            data = self._normalise_data(xscaler, data)

        # Pick feature and label data from all data
        l_data = data.loc[:,self.options.meta_params + self.options.label_params]
        f_data = data.loc[:,self.options.meta_params + self.options.feature_params]

        assert l_data.shape[0] == f_data.shape[0]

        target = l_data['delay'].astype(np.float64).values.ravel()
        features = f_data.drop(columns=['trainstation', 'time']).astype(np.float64).values

        y_pred = predictor.predict(features)

        if self.options.normalize:
            y_pred = yscaler.inverse_transform(y_pred)

        return target, y_pred

    def pred(self, times, data):
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
        else:
            return self.pred_scikit(times, data)
