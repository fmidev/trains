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

from lib.io import IO
from lib.viz import Viz
from lib.bqhandler import BQHandler
from lib import config as _config
from lib import convlstm


def main():
    """
    Main program
    """

    # Print GPU availability
    local_device_protos = device_lib.list_local_devices()
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])

    bq = BQHandler()
    io = IO(gs_bucket=options.gs_bucket)
    viz = Viz(io)

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {}.{} and time range {} - {}'.format(options.feature_dataset,
                                                                     options.feature_table,
                                                                     starttime.strftime('%Y-%m-%d'),
                                                                     endtime.strftime('%Y-%m-%d')))


    all_param_names = list(set(options.label_params + options.feature_params + options.meta_params))
    aggs = io.get_aggs_from_param_names(options.feature_params)

    logging.info('Building model...')
    dim = len(options.feature_params)
    if options.month: dim += 1
    model = convlstm.Regression(options, dim).get_model()

    logging.info('Reading data...')
    bq.set_params(batch_size=2500000,
                  loc_col='trainstation',
                  project=options.project,
                  dataset=options.feature_dataset,
                  table=options.feature_table,
                  parameters=all_param_names,
                  locations=options.train_stations,
                  only_winters=options.only_winters,
                  reason_code_table=options.reason_code_table)

    data = bq.get_rows(starttime, endtime)

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

    data.sort_values(by=['time', 'trainstation'], inplace=True)

    if options.month:
        logging.info('Adding month to the dataset...')
        data['month'] = data['time'].map(lambda x: x.month)
        options.feature_params.append('month')

    if options.normalize:
        logging.info('Normalizing data...')
        xscaler = StandardScaler()
        yscaler = StandardScaler()

        labels = data.loc[:, options.label_params].astype(np.float32).values.reshape((-1, 1))
        yscaler.fit(labels)
        scaled_labels = pd.DataFrame(yscaler.transform(labels), columns=['delay'])

        non_scaled_data = data.loc[:,options.meta_params+ ['class']]
        scaled_features = pd.DataFrame(xscaler.fit_transform(data.loc[:,options.feature_params]),
                                       columns=options.feature_params)

        data = pd.concat([non_scaled_data, scaled_features, scaled_labels], axis=1)

        fname = options.save_path+'/xscaler.pkl'
        io.save_scikit_model(xscaler, fname, fname)
        fname = options.save_path+'/yscaler.pkl'
        io.save_scikit_model(yscaler, fname, fname)

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
    batch_size = 512
    logging.info('Batch size: {}'.format(batch_size))

    # Initialization
    losses, val_losses, accs, val_accs, steps = [], [], [], [], []

    boardcb = TensorBoard(log_dir=options.log_dir+'/lstm',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

    logging.info('Data shape: {}'.format(data_train.loc[:, options.feature_params].values.shape))

    data_gen = TimeseriesGenerator(data_train.loc[:, options.feature_params].values,
                                   data_train.loc[:, options.label_params].values,
                                   length=24,
                                   sampling_rate=1,
                                   batch_size=batch_size)

    data_test_gen = TimeseriesGenerator(data_test.loc[:, options.feature_params].values,
                                        data_test.loc[:, options.label_params].values,
                                        length=24,
                                        sampling_rate=1,
                                        batch_size=batch_size)

    logging.info('X batch size: {}'.format(data_gen[0][0].shape))
    logging.info('Y batch size: {}'.format(data_gen[1][0].shape))

    history = model.fit_generator(data_gen,
                                  validation_data=data_test_gen,
                                  epochs=options.epochs,
                                  callbacks=[boardcb]) #, batch_size=64)

    history_fname = options.save_path+'/history.pkl'
    io.save_keras_model(options.save_file, history_fname, model, history.history)

    scores = model.evaluate_generator(data_test_gen)
    i = 0
    error_data = {}
    for name in model.metrics_names:
        logging.info('{}: {:.4f}'.format(name, scores[i]))
        error_data[name] = [scores[i]]
        i += 1

    fname = '{}/training_time_validation_errors.csv'.format(options.output_path)
    io.write_csv(error_data, filename=fname, ext_filename=fname)

    pred = model.predict_generator(data_test_gen)

    #io.log_class_dist(pred, 4)
    #print(history.history)
    fname = options.output_path+'/learning_over_time.png'
    viz.plot_nn_perf(history.history, metrics={'Error': {'mean_squared_error': 'MSE',
                                                         'mean_absolute_error': 'MAE'}},
                                                         filename=fname)




if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', type=str, default=None, help='Configuration file name')
    parser.add_argument('--config_name', type=str, default=None, help='Configuration file name')

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

    logging.info('Using configuration {}|{}'.format(options.config_filename,
                                                    options.config_name))

    main()
