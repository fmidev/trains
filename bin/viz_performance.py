import sys, os, argparse, logging, json
import datetime as dt
from datetime import timedelta

import itertools
from collections import OrderedDict

import math
import numpy as np

import tensorflow as tf
from tensorflow.python.estimator.export import export
from tensorflow.python.framework import constant_op
from tensorflow.contrib import predictor

from sklearn import metrics

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

    starttime, endtime = io.get_dates(options)
    logging.info('Using feature dataset {}, label dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                                            options.label_dataset,
                                                                                            starttime.strftime('%Y-%m-%d'),
                                                                                            endtime.strftime('%Y-%m-%d')))

    params, param_names = io.read_parameters(options.parameters_file, drop=2)
    calc_param_names = ['flashcount', 'max_precipitation3h', 'max_precipitation6h']
    meta_param_names = ['trainstation', 'time']

    feature_param_names = param_names + calc_param_names
    label_param_names = ['train_type', 'delay']

    all_param_names = label_param_names + feature_param_names + meta_param_names

    aggs = io.get_aggs_from_params(feature_param_names)

    if options.model_file is not None:
        predictor = io.load_scikit_model(options.model_file)
    else:
        predictor = tf.contrib.predictor.from_saved_model(options.model_path)

    avg_delay = {}
    avg_pred_delay = {}
    station_count = 0
    all_times = set()

    station_rmse = {}
    station_median_abs_err = {}
    station_r2 = {}

    stationList = io.get_train_stations(options.stations_file)

    if options.stations is not None:
        stations = options.stations.split(',')
    else:
        stations = stationList.keys()

    for station in stations:
        stationName = '{} ({})'.format(stationList[station]['name'], station)
        logging.info('Processing station {}'.format(stationName))

        data = bq.get_rows(starttime,
                           endtime,
                           loc_col='trainstation',
                           project=options.project,
                           dataset=options.feature_dataset,
                           table=options.feature_table,
                           parameters=all_param_names,
                           locations=[station])

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
        times = data.loc[:,'time']

        logging.debug('Labels shape: {}'.format(l_data.shape))
        logging.debug('Features shape: {}'.format(f_data.shape))
        logging.info('Processing {} rows...'.format(len(f_data)))

        if len(data) == 0:
            continue

        assert l_data.shape[0] == f_data.shape[0]

        station_count += 1

        target = l_data['delay'].astype(np.float64).as_matrix().ravel()
        features = f_data.drop(columns=['trainstation', 'time']).astype(np.float64).as_matrix()

        if options.model_file is not None:
            y_pred = predictor.predict(features)
        else:
            y_pred = predictor({'X': features})['pred_output'].ravel()
        #y_pred=None

        #print(times)
        i = 0
        for t in times:
            if t not in avg_delay.keys():
                avg_delay[t] = [target[i]]
                avg_pred_delay[t] = [y_pred[i]]
            else:
                avg_delay[t].append(target[i])
                avg_pred_delay[t].append(y_pred[i])
            i += 1

        all_times = all_times.union(set(times))

        rmse = math.sqrt(metrics.mean_squared_error(target, y_pred))
        median_abs_err = metrics.median_absolute_error(target, y_pred)
        r2 = metrics.r2_score(target, y_pred)
        station_rmse[station] = rmse
        station_median_abs_err[station] = median_abs_err
        station_r2[station] = r2
        logging.info('RMSE for station {}: {}'.format(stationName, rmse))
        logging.info('Median absolute error for station {}: {}'.format(median_abs_err, rmse))
        logging.info('R2 score for station {}: {}'.format(r2, rmse))

        times_formatted = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in times]
        delay_data = {'times': times_formatted, 'delay': target, 'predicted delay': y_pred}
        io.write_csv(delay_data, '{}/delays_{}.csv'.format(options.save_path, station))
        viz.plot_delay(times, target, y_pred, 'Delay for station {}'.format(stationName), '{}/{}.png'.format(options.save_path, station))

    io.dict_to_csv(station_rmse, '{}/station_rmse.csv'.format(options.save_path))
    io.dict_to_csv(station_median_abs_err, '{}/station_median_absolute_error.csv'.format(options.save_path))
    io.dict_to_csv(station_r2, '{}/station_r2.csv'.format(options.save_path))

    all_times = sorted(list(all_times))
    for t,l in avg_delay.items():
        avg_delay[t] = sum(l)/len(l)
    for t, l in avg_pred_delay.items():
        avg_pred_delay[t] = sum(l)/len(l)        
    avg_delay = list(OrderedDict(sorted(avg_delay.items(), key=lambda t: t[0])).values())
    avg_pred_delay = list(OrderedDict(sorted(avg_pred_delay.items(), key=lambda t: t[0])).values())

    rmse = math.sqrt(metrics.mean_squared_error(avg_delay, avg_pred_delay))
    median_abs_err = metrics.median_absolute_error(avg_delay, avg_pred_delay)
    r2 = metrics.r2_score(avg_delay, avg_pred_delay)
    logging.info('RMSE for average delay over all stations: {}'.format(rmse))
    logging.info('Median absolute error for average delay over all stations: {}'.format(median_abs_err))
    logging.info('R2 score for average delay over all stations: {}'.format(r2))

    all_times_formatted = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in all_times]
    delay_data = {'times': all_times_formatted, 'delay': avg_delay, 'predicted delay': avg_pred_delay}
    io.write_csv(delay_data, '{}/avg_delays_all_stations.csv'.format(options.save_path))

    viz.plot_delay(all_times, avg_delay, avg_pred_delay, 'Average delay for all station', '{}/avg_all_stations.png'.format(options.save_path))



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, default='2011-02-01', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2011-03-01', help='End time of the classification data interval')
    parser.add_argument('--save_path', type=str, default=None, help='Output path')
    parser.add_argument('--model_path', type=str, default=None, help='Path of TensorFlow Estimator file')
    parser.add_argument('--model_file', type=str, default=None, help='Path and filename of SciKit model file. If this is given, model_path is ignored.')
    parser.add_argument('--project', type=str, default='trains-197305', help='BigQuery project name')
    parser.add_argument('--feature_dataset', type=str, default='trains_all_features', help='Dataset name for features')
    parser.add_argument('--label_dataset', type=str, default='trains_labels', help='Dataset name for labels')
    parser.add_argument('--feature_table', type=str, default='features', help='Table name for features')
    parser.add_argument('--label_table', type=str, default='labels_passenger', help='Table name for labels')
    parser.add_argument('--stations', type=str, default=None, help='List of train stations separated by comma')
    parser.add_argument('--stations_file', type=str, default='cnf/stations.json', help='Stations file, list of stations to process')
    parser.add_argument('--parameters_file', type=str, default='cnf/parameters_shorten.txt', help='Param conf filename')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')
    parser.add_argument('--output_path', type=str, default=None, help='Path where visualizations are saved')

    options = parser.parse_args()

    if options.save_path is None:
        options.save_path = 'visualizations/'+options.feature_dataset

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
