import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np
import pandas as pd
from configparser import ConfigParser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import class_weight

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.python.client import device_lib

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq
from lib import config as _config
from lib import convlstm

def log_class_dist(data):
    """
    Log class distributions
    """
    c0 = sum((data < 1))
    c1 = sum((data > 0) & (data < 2))
    c2 = sum((data > 1) & (data < 3))
    c3 = sum((data > 2))
    c_all = len(data)
    logging.info('Class sizes: 0: {} ({:.02f}%), 1: {} ({:.02f}%), 2: {} ({:.02f}%), 3: {} ({:.02f}%)'.format(c0, c0/c_all*100, c1, c1/c_all*100, c2, c2/c_all*100, c3, c3/c_all*100))


def report_cv_results(results, filename=None, n_top=3):
    """
    Write cross validation results to file.
    """
    res = ""
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            res += "Model with rank: {0}\n".format(i)
            res += "Mean validation score: {0:.3f} (std: {1:.3f})\n".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate])
            res += "Parameters: {0}\n".format(results['params'][candidate])
            res += "\n"

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(res)

    logging.info(res)



def main():
    """
    Main program
    """

    local_device_protos = device_lib.list_local_devices()
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])

    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    #save_path = options.save_path+'/'+options.config_name

    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    logging.info('Building model...')
    model = convlstm.Classifier().get_model()

    logging.info('Reading data...')
    bq.set_params(starttime,
                  endtime,
                  batch_size=2500000,
                  loc_col='trainstation',
                  project=options.project,
                  dataset=options.feature_dataset,
                  table=options.feature_table,
                  parameters=all_param_names)

    data = bq.get_rows()

    data = io.filter_train_type(labels_df=data,
                                train_types=options.train_types,
                                sum_types=True,
                                train_type_column='train_type',
                                location_column='trainstation',
                                time_column='time',
                                sum_columns=['train_count','delay'],
                                aggs=aggs)

    if options.y_avg_hours is not None:
        data = io.calc_running_delay_avg(data, options.y_avg_hours)

    if options.y_avg:
        data = io.calc_delay_avg(data)

    # Classify rows
    if 'class' not in data.columns:
        data = io.classify(data)
        
    log_class_dist(data.loc[:,'class'])

    #c0 = sum((data['class'] < 1))
    #c1 = sum((data['class'] > 0) & (data['class'] < 2))
    #c2 = sum((data['class'] > 1) & (data['class'] < 3))
    #c3 = sum((data['class'] > 2))
    #c_all = len(data)
    #logging.info('Class sizes: 0: {} ({:.02f}%), 1: {} ({:.02f}%), 2: {} ({:.02f}%), 3: {} ({:.02f}%)'.format(c0, c0/c_all*100, c1, c1/c_all*100, c2, c2/c_all*100, c3, c3/c_all*100))

    data.sort_values(by=['time', 'trainstation'], inplace=True)

    if options.normalize:
        logging.info('Normalizing data...')
        xscaler = StandardScaler()

        non_scaled_data = data.loc[:,options.meta_params+ ['class']]
        scaled_features = pd.DataFrame(xscaler.fit_transform(data.loc[:,options.feature_params]),
                                       columns=options.feature_params)

        data = pd.concat([non_scaled_data, scaled_features], axis=1)

    if options.pca:
        logging.info('Doing PCA analyzis for the data...')
        ipca = IncrementalPCA(n_components=options.pca_components,
                              whiten = options.whiten,
                              copy = False)

        non_processed_data = data.loc[:,options.meta_params+ options.label_params]
        processed_data = data.loc[:, options.feature_params]
        ipca.fit(processed_data)
        processed_features = pd.DataFrame(ipca.transform(processed_data))

        data = pd.concat([non_processed_data, processed_data], axis=1)

        fname = options.output_path+'/ipca_explained_variance.png'
        viz.explained_variance(ipca, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)

    data_train, data_test = train_test_split(data, test_size=0.33)

    # Define model
    batch_size = io.get_batch_size(data_train, options.pad_strategy, quantile=options.quantile)
    batch_size = 512
    logging.info('Batch size: {}'.format(batch_size))

    # Initialization
    losses, val_losses, accs, val_accs, steps = [], [], [], [], []

    boardcb = TensorBoard(log_dir=options.log_dir,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

    logging.info('Data shape: {}'.format(data_train.loc[:, options.feature_params].values.shape))
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(data_train.loc[:, 'class'].values),
                                                      data_train.loc[:, 'class'].values)
    weights = {}
    i = 0
    for w in class_weights:
        weights[i] = w
        i += 1

    logging.info('Class weights: {}'.format(weights))

    data_gen = TimeseriesGenerator(data_train.loc[:, options.feature_params].values,
                                   to_categorical(data_train.loc[:, 'class'].values),
                                   length=24,
                                   sampling_rate=1,
                                   batch_size=batch_size)

    data_test_gen = TimeseriesGenerator(data_test.loc[:, options.feature_params].values,
                                        to_categorical(data_test.loc[:, 'class'].values),
                                        length=24,
                                        sampling_rate=1,
                                        batch_size=batch_size)

    logging.info('X batch size: {}'.format(data_gen[0][0].shape))
    logging.info('Y batch size: {}'.format(data_gen[1][0].shape))

    history = model.fit_generator(data_gen,
                                  validation_data=data_test_gen,
                                  epochs=3,
                                  class_weight=class_weights,
                                  callbacks=[boardcb]) #, batch_size=64)

    model_fname = options.save_path+'/model.json'
    weights_fname = options.save_path+'/weights.h5'
    history_fname = options.save_path+'/history.pkl'
    io.save_model(model_fname, weights_fname, history_fname, model, history.history)

    scores = model.evaluate_generator(data_test_gen)
    i = 0
    error_data = {}
    for name in model.metrics_names:
        logging.info('{}: {:.4f}'.format(name, scores[i]))
        error_data[name] = [scores[i]]
        i += 1

    fname = '{}/training_time_validation_errors.csv'.format(options.output_path)
    io.write_csv(error_data, filename=fname, ext_filename=fname)

    pred_proba = model.predict_generator(data_test_gen)
    pred = np.argmax(pred_proba, axis=1)

    log_class_dist(pred)
    #print(history.history)
    fname = options.output_path+'/learning_over_time.png'
    viz.plot_nn_perf(history.history, metrics={'[%]': {'acc': 'Accuracy',
                                                       'F1': 'F1 Score',
                                                       'Precision': 'Precision',
                                                       'Recall': 'Recall'}},
                     filename=fname)

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', type=str, default=None, help='Configuration file name')
    parser.add_argument('--config_name', type=str, default=None, help='Configuration file name')
    parser.add_argument('--dev', type=int, default=0, help='1 for development mode')

    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    options = parser.parse_args()

    _config.read(options)

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
