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

class Predictor():

    def __init__(self, io, model_loader, options):
        self.io = io
        self.model_loader = model_loader
        self.options = options

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
            fname=self.options.save_path+'/yscaler.pkl'
            self.io._download_from_bucket(fname, fname, force=True)
            yscaler = self.io.load_scikit_model(fname)

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

        # Pick feature and label data from all data
        l_data = data.loc[:,self.options.meta_params + self.options.label_params]
        f_data = data.loc[:,self.options.meta_params + self.options.feature_params]

        assert l_data.shape[0] == f_data.shape[0]

        target = l_data['delay'].astype(np.float64).values.ravel()
        features = f_data.drop(columns=['trainstation', 'time']).astype(np.float64).values

        y_pred = predictor.predict(features)

        return target, y_pred

    def pred(self, times, data):
        """
        Run SciKit prediction

        times : lst
                list of time steps to which prediction is to be done
        data : DataFrame
               feature and target data (got from bq.get_rows() and possibly
               filtered by filter_train_type(), sorted by time and trainstation )

        return lst target, lst prediction
        """
        if self.options.tf:
            return self.pred_tf(times, data)
        else:
            return self.pred_scikit(times, data)
