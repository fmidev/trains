import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np
import tensorflow as tf
#from tensorflow.python.estimator.export import export
from tensorflow.python.estimator import exporter
from tensorflow.python.framework import constant_op

from sklearn.model_selection import train_test_split

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq

from lib.model_functions import lr
from lib.model_functions import rf
from lib.model_functions import general


def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()
    io = _io.IO()
    viz = _viz.Viz()

    if not os.path.exists(options.save_path):
        os.makedirs(options.save_path)

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    model_filename = options.save_path+'/model_state.ckpt'
    export_dir = options.save_path+'/serving'

    starttime, endtime = io.get_dates(options)
    logging.info('Using feature dataset {}, label dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                                            options.label_dataset,
                                                                                            starttime.strftime('%Y-%m-%d'),
                                                                                            endtime.strftime('%Y-%m-%d')))

    #locations = a.get_locations_by_dataset('trains-1.0', starttime, endtime)

    params, param_names = io.read_parameters(options.parameters_file, drop=2)
    calc_param_names = ['flashcount', 'max_precipitation3h', 'max_precipitation6h']
    meta_param_names = ['trainstation', 'time']

    feature_param_names = param_names + calc_param_names
    label_param_names = ['train_type', 'delay']

    all_param_names = label_param_names + feature_param_names + meta_param_names

    aggs = io.get_aggs_from_params(feature_param_names)

    #model_params{n_dim = len(feature_param_names)
    # Define number of gradient descent loops

    if True:
        params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_classes=1,
        num_features=len(feature_param_names),
        regression=True,
        num_trees=50,
        max_nodes=1000
        ).fill()

        model = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(        model_dir=options.log_dir,
        params=params)

        run_config = tf.estimator.RunConfig(save_summary_steps=None,
                                            save_checkpoints_secs=None)
        model = tf.estimator.Estimator(model_fn=rf.model_func,
                                       model_dir=options.log_dir,
                                       params={'n_dim': len(feature_param_names)},
                                       config=run_config
                                       )
    else:
        model = tf.estimator.Estimator(model_fn=lr.model_func,
                                       model_dir=options.log_dir,
                                       params={'n_dim': len(feature_param_names)}
                                       )

    start = starttime
    end = start + timedelta(days=options.day_step, hours=options.hour_step)
    if end > endtime: end = endtime

    while end <= endtime:
        logging.info('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'),
                                                            end.strftime('%Y-%m-%d %H:%M')))

        try:
            # l_data = bq.get_rows(start,
            #                      end,
            #                      loc_col='loc_name',
            #                      project=options.project,
            #                      dataset=options.label_dataset,
            #                      table=options.label_table)


            data = bq.get_rows(start,
                               end,
                               loc_col='trainstation',
                               project=options.project,
                               dataset=options.feature_dataset,
                               table=options.feature_table,
                               parameters=all_param_names)

            data = io.filter_train_type(labels_df=data,
                                        train_types=['K','L'],
                                        sum_types=True,
                                        train_type_column='train_type',
                                        location_column='trainstation',
                                        time_column='time',
                                        sum_columns=['delay'],
                                        aggs=aggs)

            data.sort_values(by=['time', 'trainstation'], inplace=True)
            l_data = data.loc[:,meta_param_names + label_param_names]
            f_data = data[meta_param_names + feature_param_names]

            #print(l_data)
            #print(f_data)

        except ValueError as e:
            f_data, l_data = [], []

        if len(f_data) == 0 or len(l_data) == 0:
            start = end
            end = start + timedelta(days=options.day_step, hours=options.hour_step)
            continue

        # l_data = io.filter_train_type(labels_df=l_data,
        #                               traintypes=[0,1],
        #                               train_type_column='train_type',
        #                               location_column='loc_id',
        #                               sum_columns=['late_minutes','train_count','total_late_minutes'],
        #                               sum_types=True)
        f_data.rename(columns={'trainstation':'loc_name'}, inplace=True)
        # l_data, f_data = io.filter_labels(l_data,
        #                                   f_data,
        #                                   location_column='loc_name',
        #                                   uniq=True) #, invert=True)

        logging.debug('Labels shape: {}'.format(l_data.shape))
        logging.debug('Features shape: {}'.format(f_data.shape))
        logging.info('Processing {} rows...'.format(len(f_data)))

        assert l_data.shape[0] == f_data.shape[0]

        target = l_data['delay'].astype(np.float32).values
        features = f_data.drop(columns=['loc_name', 'time']).astype(np.float32).values
        # logging.debug('Data types of features: {}'.format(f_data.dtypes))

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33)

        # print(X_train[0:10])
        # print(y_train[0:10])
        n_samples, n_dims = X_train.shape
        #    saver = tf.train.Saver()

        # Select random mini-batch
        def input_train_rf():
            indices = np.random.choice(n_samples, options.batch_size)
            X_batch, y_batch = X_train[indices], y_train[indices]
            train_input_fn = tf.estimator.inputs.numpy_input_fn(x= X_batch,
                                                                y= y_batch,
                                                                shuffle=True)
            x, y = train_input_fn()
            return x, y

        def input_test_rf():
            indices = np.random.choice(len(X_test), options.batch_size)
            X_batch, y_batch = X_test[indices], y_test[indices]
            train_input_fn = tf.estimator.inputs.numpy_input_fn(x= X_batch,
                                                                y= y_batch,
                                                                shuffle=True)
            x, y = train_input_fn()
            return x, y

        def input_train():
            indices = np.random.choice(n_samples, options.batch_size)
            X_batch, y_batch = X_train[indices], y_train[indices]
            return X_batch, y_batch

        def input_test():
            indices = np.random.choice(len(X_test), options.batch_size)
            X_batch, y_batch = X_test[indices], y_test[indices]
            return X_batch, y_batch

        print('Training')
        print(model.train(input_fn=input_train_rf, steps=options.n_loops))
        print('Evaluating')
        model.evaluate(input_fn=input_test, steps=1)

        start = end
        end = start + timedelta(days=options.day_step, hours=options.hour_step)

    model.export_savedmodel(
        export_dir,
        rf.serving_input_receiver_fn
    )

    export(
        model,
        export_dir,
        rf.serving_input_receiver_fn
    )

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, help='End time of the classification data interval')
    parser.add_argument('--save_path', type=str, default=None, help='Model save path and filename')
    parser.add_argument('--model', type=str, default='rf', help='Model save path and filename')
    parser.add_argument('--project', type=str, default='trains-197305', help='BigQuery project name')
    parser.add_argument('--feature_dataset', type=str, default='trains_all_features', help='Dataset name for features')
    parser.add_argument('--label_dataset', type=str, default='trains_labels', help='Dataset name for labels')
    parser.add_argument('--feature_table', type=str, default='features', help='Table name for features')
    parser.add_argument('--label_table', type=str, default='labels_passenger', help='Table name for labels')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--day_step', type=int, default=30, help='How many days are handled in one step')
    parser.add_argument('--hour_step', type=int, default=0, help='How many hours are handled in one step')

    #parser.add_argument('--stations_file', type=str, default=None, help='Stations file to rename stations from loc_id to station name')
    parser.add_argument('--parameters_file', type=str, default='cnf/parameters_shorten.txt', help='Param conf filename')
    parser.add_argument('--log_dir', type=str, default=None, help='Tensorboard log dir')
    parser.add_argument('--dev', type=int, default=0, help='1 for development mode')
    parser.add_argument('--db_config_file',
                        type=str,
                        default=None,
                        help='GS address for db config file (if none, ~/.mlfdbconfig is used)')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')
    parser.add_argument('--output_path', type=str, default=None, help='Path where visualizations are saved')

    options = parser.parse_args()

    if options.save_path is None:
        options.save_path = 'models/'+options.model+'/'+options.feature_dataset

    if options.output_path is None:
        options.output_path = 'results/'+options.model+'/'+options.feature_dataset

    if options.log_dir is None:
        options.log_dir = '/tmp/'+options.model+'/'+options.feature_dataset

    if options.dev == 1: options.n_loops = 100
    else: options.n_loops = 10000

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
