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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.contrib import rnn

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq
from lib import config as _config
from lib import LSTM

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

def get_batch_size(data, pad_strategy='pad', quantile=None):
    """
    Get batch size based on pad_strategy. See extract_batch docs for more info.
    """
    if pad_strategy == 'sample':
        return min(data.groupby(['time']).size())
    elif pad_strategy == 'pad':
        return max(data.groupby(['time']).size())
    elif pad_strategy == 'drop':
        #logging.debug(data.groupby(['time']).size())
        return int(data.groupby(['time']).size().quantile(quantile))

def pad_along_axis(a, target_length, constant_values=-99, axis=0):
    """
    Pad along given axis
    """
    pad_size = target_length - a.shape[axis]
    axis_nb = len(a.shape)

    if pad_size < 0:
        return a

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(a, pad_width=npad, mode='constant', constant_values=constant_values)

    return b

def extract_batch(data, n_timesteps, batch_num=0, batch_size=None, pad_strategy='pad', quantile=None):
    """
    Extract and/or preprocess batch from data.

    data : pd.DataFrame
           Data
    n_timesteps : int
                  Number of timesteps to be used in LSTM. If <0 all timesteps are used-
    batch_num : int
                Batch number, default 0
    batch_size : int or None
                 If None, batch size is got based on pad_strategy.
                 If <0, all valid values are returned (used specially for test data)
                 If >0, given batch size is used.
    pad_strategy : str
                   - If 'pad', largest timestep (with most stations) is chosen
                   from the data and other timesteps are padded with -99
                   - If 'sample', smallest timestep is chosen and other timesteps
                   are sampled to match smallest one.
                   - If 'drop', batch_size is chosen to to match options.quantile
                   value. If for example options.quantile=0.3 70 percent of timesteps
                   are taken into account.
    quantile : float or None
               Used to drop given fraction of smaller timesteps from data if
               pad_strategy='drop'

    return : (Values shape: (n_timesteps, batch_size, n_features), Labels shape (n_timesteps, batch_size))
    """
    # Detect batch size
    if batch_size is None:
        batch_size = get_batch_size(data, pad_strategy, quantile)
    elif batch_size < 0:
        batch_size = len(data)

    # If pad_strategy is drop, drop timesteps with too few stations
    if pad_strategy == 'drop':
        t_size = data.groupby(['time']).size()
        t_size = t_size[t_size >= batch_size].index.values
        data = data[data.time.isin(t_size)]

    all_times = data.time.unique()
    stations = data.trainstation.unique()

    if n_timesteps < 0:
        n_timesteps = len(all_times)

    # Pick times for the batch
    start = batch_num*n_timesteps
    end = start + n_timesteps
    times = all_times[start:end]

    values = []
    labels = []

    # Go through times
    for t in times:
        # Pick data for current time
        timestep_values = data[data.loc[:,'time'] == t]

        # If pad_strategy is drop and timestep is too small, ignore it and continue
        if pad_strategy == 'drop' and batch_size > len(timestep_values):
            continue

        # If pad_strategy is sample or drop, sample data to match desired batch_size
        if pad_strategy in ['sample','drop'] and batch_size < len(timestep_values):
            timestep_values = timestep_values.sample(batch_size)

        # pd dataframe to np array
        timestamp_values = timestep_values.drop(columns=['time', 'trainstation', 'delay', 'train_type']).astype(np.float32).values
        label_values = timestep_values.loc[:,'delay'].astype(np.float32).values

        # if pad_strategy is pad and timestep is too small, pad it with -99
        if pad_strategy == 'pad' and batch_size > len(timestamp_values):
            timestamp_values = pad_along_axis(timestamp_values, batch_size, -99)
            label_values = pad_along_axis(label_values, batch_size, -99)

        values.append(timestamp_values)
        labels.append(label_values)

    values = np.array(values)
    labels = np.array(labels)
    #values = np.rollaxis(np.array(values), 1)
    #labels = np.reshape(np.rollaxis(np.array(labels), 1), (batch_size, n_timesteps, 1))

    logging.debug('Values shape: {}'.format(values.shape))
    logging.debug('Labels shape: {}'.format(labels.shape))

    return values, labels

def main():
    """
    Main program
    """

    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    save_path = options.save_path+'/'+options.config_name

    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    # Initialise errors
    rmses, maes, steps = [], [], []

    # Define model
    model = LSTM.LSTM(options.time_steps, len(options.feature_params), 1, options.n_hidden, options.lr)

    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(options.log_dir, graph=tf.get_default_graph())
    tf.summary.scalar('MSE', model.loss)
    tf.summary.scalar('RMSE', model.rmse)
    tf.summary.scalar('MAE', model.mae)
    tf.summary.histogram('y_pred_hist', model.y_pred)
    merged_summary_op = tf.summary.merge_all()

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
                                sum_columns=['delay'],
                                aggs=aggs)
    data.sort_values(by=['time', 'trainstation'], inplace=True)

    data_train, data_test = train_test_split(data, test_size=0.33)
    X_test, y_test = extract_batch(data_test, options.time_steps, batch_size=None, pad_strategy=options.pad_strategy, quantile=options.quantile)

    # Batch size just for information
    batch_size = get_batch_size(data_train, options.pad_strategy, quantile=options.quantile)
    logging.info('Using batch size: {}'.format(batch_size))

    train_step = 0
    while True:
        X_train, y_train = extract_batch(data_train, options.time_steps, train_step, pad_strategy=options.pad_strategy, quantile=options.quantile)

        if(len(X_train) < options.time_steps):
            break

        if options.cv:
            logging.info('Doing random search for hyper parameters...')

            param_grid = {"C": [0.001, 0.01, 0.1, 1, 10],
                          "epsilon": [0.01, 0.1, 0.5],
                          "kernel": ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
                          "degree": [2, 3 ,4],
                          "shrinking": [True, False],
                          "gamma": [0.001, 0.01, 0.1],
                          "coef0": [0, 0.1, 1]}

            random_search = RandomizedSearchCV(model,
                                               param_distributions=param_grid,
                                               n_iter=int(options.n_iter_search),
                                               n_jobs=-1)

            random_search.fit(X_train, y_train)
            logging.info("RandomizedSearchCV done.")
            fname = options.output_path+'/random_search_cv_results.txt'
            report_cv_results(random_search.cv_results_, fname)
            io._upload_to_bucket(filename=fname, ext_filename=fname)
            sys.exit()
        else:
            if train_step == 0:
                logging.info('Training...')
            feed_dict = {model.X: X_train,
                         model.y: y_train}

            _, loss, pred = sess.run(
                [model.train_op, model.loss, model.pred],
                feed_dict=feed_dict)

        # Metrics
        feed_dict = {model.X: X_test,
                     model.y: y_test}
                     #model.cell_init_state: state}

        val_loss, rmse, mae, y_pred, summary = sess.run(
            [model.loss, model.rmse, model.mae, model.y_pred, merged_summary_op],
            feed_dict=feed_dict)

        #print(y_pred)
        #print(y_test)
        #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        #mae = mean_absolute_error(y_test, y_pred)

        rmses.append(rmse)
        maes.append(mae)
        steps.append(train_step)

        summary_writer.add_summary(summary, train_step * batch_size)
        if train_step%50 == 0:
            logging.info("Step {}:".format(train_step))
            logging.info("Training loss: {:.4f}".format(loss))
            logging.info("Validation MSE: {:.4f}".format(val_loss))
            logging.info('Validation RMSE: {}'.format(rmse))
            logging.info('Validation MAE: {}'.format(mae))
            logging.info('................')
            saver.save(sess, save_path)

        train_step += 1
        # <-- while True:

    saver.save(sess, save_path)
    #io.save_scikit_model(model, filename=options.save_file, ext_filename=options.save_file)

    try:
        fname = options.output_path+'/learning_over_time.png'
        metrics = [{'metrics':[{'values': rmses, 'label': 'RMSE'}],'y_label': 'RMSE'},
                   {'metrics':[{'values': maes, 'label': 'MAE'}], 'y_label': 'MAE'}]
        viz.plot_learning(metrics, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)
    except Exception as e:
        logging.error(e)

    error_data = {'steps': steps,
                  'rmse': rmses,
                  'mae': maes}
    fname = '{}/training_time_validation_errors.csv'.format(options.output_path)
    io.write_csv(error_data, filename=fname, ext_filename=fname)






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
