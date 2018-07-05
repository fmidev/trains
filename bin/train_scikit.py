import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq



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

    # steps=options.n_loops
    if options.model == 'rf':
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1)

    start = starttime
    end = start + timedelta(days=options.day_step, hours=options.hour_step)
    if end > endtime: end = endtime

    while end <= endtime:
        logging.info('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'),
                                                            end.strftime('%Y-%m-%d %H:%M')))

        try:
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

        f_data.rename(columns={'trainstation':'loc_name'}, inplace=True)

        logging.debug('Labels shape: {}'.format(l_data.shape))
        logging.debug('Features shape: {}'.format(f_data.shape))
        logging.info('Processing {} rows...'.format(len(f_data)))

        assert l_data.shape[0] == f_data.shape[0]

        target = l_data['delay'].astype(np.float32).values
        features = f_data.drop(columns=['loc_name', 'time']).astype(np.float32).values

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33)

        # print(X_train[0:10])
        # print(y_train[0:10])
        n_samples, n_dims = X_train.shape

        # indices = np.random.choice(n_samples, options.batch_size)
        # X_batch, y_batch = X_train[indices], y_train[indices]

        logging.info('Training...')
        model.fit(X_train, y_train)

        # Metrics
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if options.model == 'rf':
            logging.info('R2 score for training: {}'.format(model.score(X_train, y_train)))

        logging.info('RMSE: {}'.format(rmse))
        logging.info('MAE: {}'.format(mae))
        logging.info('R2 score: {}'.format(r2))

        start = end
        end = start + timedelta(days=options.day_step, hours=options.hour_step)

    io.save_scikit_model(model, options.save_file)


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

    options.save_file = options.save_path+'/model.pkl'

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
