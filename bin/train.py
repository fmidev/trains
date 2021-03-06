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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.client import device_lib

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

    logging.info('Reading data...')
    bq.set_params(starttime,
                  endtime,
                  batch_size=2500000,
                  loc_col='trainstation',
                  project=options.project,
                  dataset=options.feature_dataset,
                  table=options.feature_table,
                  parameters=all_param_names,
                  only_winters=options.only_winters)

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

    data.sort_values(by=['time', 'trainstation'], inplace=True)

    if options.normalize:
        logging.info('Normalizing data...')
        xscaler = StandardScaler()
        yscaler = StandardScaler()

        non_scaled_data = data.loc[:,options.meta_params]
        labels = data.loc[:, options.label_params].astype(np.float32).values.reshape((-1, 1))

        yscaler.fit(labels)
        scaled_labels = pd.DataFrame(yscaler.transform(labels), columns=['delay'])
        scaled_features = pd.DataFrame(xscaler.fit_transform(data.loc[:,options.feature_params].astype(np.float32)),
                                       columns=options.feature_params)

        data = pd.concat([non_scaled_data, scaled_features, scaled_labels], axis=1)

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
    X_test, y_test = io.extract_batch(data_test,
                                      options.time_steps,
                                      batch_size=None,
                                      pad_strategy=options.pad_strategy,
                                      quantile=options.quantile,
                                      label_params=options.label_params,
                                      feature_params=options.feature_params)

    # Define model
    batch_size = io.get_batch_size(data_train, options.pad_strategy, quantile=options.quantile)
    logging.info('Batch size: {}'.format(batch_size))
    model = LSTM.LSTM(options.time_steps, len(options.feature_params), 1, options.n_hidden, options.lr, options.p_drop, batch_size=batch_size)

    # Initialization
    rmses, mses, maes, steps, train_mse = [], [], [], [], []
    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(options.log_dir, graph=tf.get_default_graph())

    #tf.summary.scalar('Training MSE', model.loss)
    tf.summary.scalar('Validation_MSE', model.mse)
    tf.summary.scalar('Validation_RMSE', model.rmse)
    tf.summary.scalar('Validation_MAE', model.mae)
    tf.summary.histogram('y_pred_hist', model.y_pred)
    merged_summary_op = tf.summary.merge_all()
    train_summary_op = tf.summary.scalar('Training_MSE', model.loss)

    train_step = 0
    start = 0
    while True:
        # If slow is set, go forward one time step at time,
        # else proceed whole batch size
        if options.slow:
            X_train, y_train = io.extract_batch(data_train,
                                                options.time_steps,
                                                start=start,
                                                pad_strategy=options.pad_strategy,
                                                quantile=options.quantile,
                                                label_params=options.label_params,
                                                feature_params=options.feature_params)
        else:
            X_train, y_train = io.extract_batch(data_train,
                                                options.time_steps,
                                                train_step,
                                                pad_strategy=options.pad_strategy,
                                                quantile=options.quantile,
                                                label_params=options.label_params,
                                                feature_params=options.feature_params)

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
            _, loss, train_summary = sess.run(
                [model.train_op, model.loss, train_summary_op],
                feed_dict=feed_dict)

            summary_writer.add_summary(train_summary, train_step * batch_size)

        # Metrics
        feed_dict = {model.X: X_test,
                     model.y: y_test}
                     #model.cell_init_state: state}

        val_loss, rmse, mse, mae, y_pred, summary = sess.run(
            [model.loss, model.rmse, model.mse, model.mae, model.y_pred, merged_summary_op],
            feed_dict=feed_dict)

        train_mse.append(loss)
        mses.append(mse)
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
            saver.save(sess, options.save_file)

        train_step += 1
        start += 1
        # <-- while True:

    saver.save(sess, options.save_file)
    if options.normalize:
        fname = options.save_path+'/yscaler.pkl'
        io.save_scikit_model(yscaler, fname, fname)
    io._upload_dir_to_bucket(options.save_path, options.save_path)

    try:
        fname = options.output_path+'/learning_over_time.png'
        metrics = [{'metrics':[{'values': mses, 'label': 'Validation MSE'}, {'values': train_mse, 'label': 'Train MSE'}],'y_label': 'MSE'},
                   {'metrics':[{'values': rmses, 'label': 'Validation RMSE'}],'y_label': 'RMSE'},
                   {'metrics':[{'values': maes, 'label': 'Validation MAE'}], 'y_label': 'MAE'}]
        viz.plot_learning(metrics, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)
    except Exception as e:
        logging.error(e)

    error_data = {'steps': steps,
                  'mse' : mses,
                  'rmse': rmses,
                  'mae': maes,
                  'train_mse': train_mse}
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
