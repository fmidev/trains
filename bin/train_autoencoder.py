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

from keras.callbacks import TensorBoard, ModelCheckpoint

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

def get_reconst_error(model, data_x, data_y, errors, name):
    """
    Calculate and add reconstruction error to the list
    """
    error_df = pd.DataFrame({'Reconstruction_error': get_mse(model, data_x),
                             'True_class': data_y.ravel()
                             })
    print(error_df.describe())
    errors[name] = error_df
    return errors

def get_mse(model, data):
    """
    Get MSE for prediction
    """
    return np.mean(np.power(data - model.predict(data), 2), axis=1)

def main():
    """
    Main program
    """

    local_device_protos = device_lib.list_local_devices()
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])

    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz(io)

    starttime, endtime = io.get_dates(options)
    #save_path = options.save_path+'/'+options.config_name

    logging.info('Using dataset {}.{} and time range {} - {}'.format(options.feature_dataset,
                                                                     options.feature_table,
                                                                     starttime.strftime('%Y-%m-%d'),
                                                                     endtime.strftime('%Y-%m-%d')))


    all_param_names = list(set(options.label_params + options.feature_params + options.meta_params))
    aggs = io.get_aggs_from_param_names(options.feature_params)

    logging.info('Reading data...')
    bq.set_params(batch_size=2500000,
                  loc_col='trainstation',
                  project=options.project,
                  dataset=options.feature_dataset,
                  table=options.feature_table,
                  parameters=all_param_names,
                  only_winters=options.only_winters)

    data = bq.get_rows(starttime, endtime)

    data = io.filter_train_type(labels_df=data,
                                train_types=options.train_types,
                                sum_types=True,
                                train_type_column='train_type',
                                location_column='trainstation',
                                time_column='time',
                                sum_columns=[],
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
        scaled_labels = pd.DataFrame(yscaler.fit_transform(labels), columns=['delay'])

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

    # Divide data to normal and delayed cases
    data_test = data[(data.loc[:, 'class'] >= options.class_limit)]
    data = data[(data.loc[:, 'class'] < options.class_limit)]

    data_train, data_val = train_test_split(data, test_size=0.33)
    data_train_x = data_train.loc[:, options.feature_params].values
    data_train_y = data_train.loc[:, options.label_params].values
    data_val_x = data_val.loc[:, options.feature_params].values
    data_val_y = data_val.loc[:, options.label_params].values

    # Initialization
    logging.info('Building model...')
    model = convlstm.Autoencoder(data_train_x.shape[1]).get_model()

    losses, val_losses, accs, val_accs, steps = [], [], [], [], []

    boardcb = TensorBoard(log_dir=options.log_dir,
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

    logging.info('Data shape: {}'.format(data_train.loc[:, options.feature_params].values.shape))

    history = model.fit(data_train_x, data_train_x,
                        validation_data=(data_val_x, data_val_x),
                        epochs=3,
                        callbacks=[boardcb]) #, batch_size=64)

    history_fname = options.save_path+'/history.pkl'
    io.save_keras_model(options.save_file, history_fname, model, history.history)

    # Reconstruction errors
    logging.info('Plotting reconstruction errors...')

    errors = {}
    logging.info('Train:')
    errors = get_reconst_error(model, data_train_x, data_train_y.ravel(), errors, 'Train')

    logging.info('Validation:')
    errors = get_reconst_error(model, data_val_x, data_val_y.ravel(), errors, 'Validation')

    logging.info('Test:')
    data_test_x = data_test.loc[:, options.feature_params].values
    data_test_y = data_test.loc[:, options.label_params].values

    errors = get_reconst_error(model, data_test_x, data_test_y.ravel(), errors, 'Test')

    for i in np.arange(4):
        fname = options.output_path+'/reconstruction_error_{}.png'.format(i)
        viz.reconstruction_error(errors, desired_class=i, filename=fname)

    fname = options.output_path+'/reconstruction_error_all.png'.format(i)
    viz.reconstruction_error(errors, filename=fname)


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
