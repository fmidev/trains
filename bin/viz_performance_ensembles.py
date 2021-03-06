import sys, os, argparse, logging, json
import datetime as dt
from datetime import timedelta

import itertools
from collections import OrderedDict

import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.estimator.export import export
from tensorflow.python.framework import constant_op
from tensorflow.contrib import predictor

from sklearn import metrics

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq
from lib import config as _config

def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    # Get params
    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    io._download_from_bucket(options.save_file, options.save_file)
    logging.info('Loadung model from {}...'.format(options.save_file))
    predictor = io.load_scikit_model(options.save_file)

    # Init error dicts
    avg_delay = {}
    avg_pred_delay = {}
    station_count = 0
    all_times = set()

    station_rmse = {}
    station_median_abs_err = {}
    station_r2 = {}

    # If stations are given as argument use them, else use all stations
    logging.info('Loading stations from {}...'.format(options.stations_file))
    stationList = io.get_train_stations(options.stations_file)
    if options.stations is not None:
        stations = options.stations.split(',')
    else:
        stations = stationList.keys()

    # Get data
    #stationName = '{} ({})'.format(stationList[station]['name'], station)
    #logging.info('Processing station {}'.format(stationName))

    # Read data and filter desired train types (ic and commuter)
    logging.info('Loading data...')
    data = bq.get_rows(starttime,
                       endtime,
                       loc_col='trainstation',
                       project=options.project,
                       dataset='trains_testset',
                       table='features_1',
                       parameters=all_param_names,
                       locations=stations)

    data = io.filter_train_type(labels_df=data,
                                train_types=['K','L'],
                                sum_types=True,
                                train_type_column='train_type',
                                location_column='trainstation',
                                time_column='time',
                                sum_columns=['delay'],
                                aggs=aggs)

    assert len(data) > 0, "Empty data"

    if options.y_avg_hours is not None:
        data = io.calc_running_delay_avg(data, options.y_avg_hours)

    data.sort_values(by=['time', 'trainstation'], inplace=True)
    logging.info('Processing {} rows...'.format(len(data)))

    # Pick times for creating error time series
    all_times = data.loc[:,'time'].unique()
    #station_count += 1

    # Pick feature and label data from all data
    l_data = data.loc[:,options.meta_params + options.label_params]
    f_data = data.loc[:,options.meta_params + options.feature_params]

    target = l_data['delay'].astype(np.float64).values.ravel()
    features = f_data.drop(columns=['trainstation', 'time']).astype(np.float64).values

    # Get data
    logging.info('Predicting...')
    y_pred = predictor.predict(features)

    # Calculate quantiles
    logging.info('Calculating fractiles...')
    groups, avg, pred = io.pred_fractiles(l_data, y_pred, stationList)

    # Go through stations
    for station in stations:

        data = pred.loc[pred['trainstation'] == station, :]
        times = data.loc[:,'time']

        if len(data) < 1:
            continue

        group = pred.loc[pred['trainstation'] == station, 'group'].values[0]
        stationName = '{} ({} | Group {})'.format(stationList[station]['name'],
                                                  station,
                                                  group)

        logging.info('Processing station {} (having {} rows)...'.format(station, len(data)))

        logging.info('Calculating errors for given station...')
        rmse = math.sqrt(metrics.mean_squared_error(data.loc[:,'delay'], data.loc[:,'pred_delay']))
        median_abs_err = metrics.median_absolute_error(data.loc[:,'delay'], data.loc[:,'pred_delay'])
        r2 = metrics.r2_score(data.loc[:,'delay'], data.loc[:,'pred_delay'])

        # Put errors to timeseries
        station_rmse[station] = rmse
        station_median_abs_err[station] = median_abs_err
        station_r2[station] = r2

        logging.info('RMSE for station {}: {}'.format(stationName, rmse))
        logging.info('Mean absolute error for station {}: {}'.format(stationName, median_abs_err))
        logging.info('R2 score for station {}: {}'.format(stationName, r2))

        # Create csv and upload it to pucket
        times_formatted = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in times]
        delay_data = {'times': times_formatted,
                      'delay': data.loc[:,'delay'].values,
                      'predicted delay': data.loc[:,'pred_delay'].values,
                      'low': data.loc[:,'pred_delay_low'].values,
                      'high': data.loc[:,'pred_delay_high'].values
                      }
        fname = '{}/delays_{}.csv'.format(options.vis_path, station)
        io.write_csv(delay_data, fname, fname)

        # Draw visualisation
        fname='{}/{}.png'.format(options.vis_path, station)
        viz.plot_delay(times,
                       data.loc[:,'delay'].values,
                       data.loc[:,'pred_delay'].values,
                       'Delay for station {}'.format(stationName),
                       fname,
                       data.loc[:,'pred_delay_low'].values,
                       data.loc[:, 'pred_delay_high'].values)
        io._upload_to_bucket(filename=fname, ext_filename=fname)

    # Save all station related results to csv and upload them to bucket
    fname = '{}/station_rmse.csv'.format(options.vis_path)
    io.dict_to_csv(station_rmse, fname, fname)
    fname = '{}/station_median_absolute_error.csv'.format(options.vis_path)
    io.dict_to_csv(station_median_abs_err, fname, fname)
    fname = '{}/station_r2.csv'.format(options.vis_path)
    io.dict_to_csv(station_r2, fname, fname)

    # Create timeseries of avg actual delay and predicted delay
    all_times = sorted(list(all_times))
    avg_delay = avg.loc[:,'avg_delay'].dropna().values.ravel()
    avg_pred_delay = avg.loc[:,'avg_pred_delay'].dropna().values.ravel()

    # Calculate average over all times and stations
    rmse = math.sqrt(metrics.mean_squared_error(avg_delay, avg_pred_delay))
    median_abs_err = metrics.median_absolute_error(avg_delay, avg_pred_delay)
    r2 = metrics.r2_score(avg_delay, avg_pred_delay)

    logging.info('RMSE for average delay over all stations: {}'.format(rmse))
    logging.info('Mean absolute error for average delay over all stations: {}'.format(median_abs_err))
    logging.info('R2 score for average delay over all stations: {}'.format(r2))

    # Write average data into file
    avg_errors = {'rmse': rmse, 'mae': median_abs_err, 'r2': r2, 'nro_of_samples': len(avg_delay)}
    fname = '{}/avg_erros.csv'.format(options.vis_path)
    io.dict_to_csv(avg_errors, fname, fname)

    # Create timeseries of average delay and predicted delays over all stations
    all_times_formatted = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in all_times]
    delay_data = {'times': all_times_formatted, 'delay': avg_delay, 'predicted delay': avg_pred_delay}

    # write csv
    fname='{}/avg_delays_all_stations.csv'.format(options.vis_path)
    io.write_csv(delay_data, fname, fname)

    for i in np.arange(0,3):
        fname='{}/avg_group_{}.png'.format(options.vis_path, (i+1))
        times = groups[i].index.values
        if len(times) < 2:
            continue

        g_avg_delay = groups[i].loc[:, 'avg_delay'].values.ravel()
        g_avg_pred_delay = groups[i].loc[:, 'avg_pred_delay'].values.ravel()
        g_avg_pred_delay_low = groups[i].loc[:, 'avg_pred_delay_low'].values.ravel()
        g_avg_pred_delay_high =  groups[i].loc[:, 'avg_pred_delay_high'].values.ravel()

        viz.plot_delay(times,
                       g_avg_delay,
                       g_avg_pred_delay,
                       'Average delay for group {}'.format(i+1),
                       fname,
                       g_avg_pred_delay_low,
                       g_avg_pred_delay_high)
        io._upload_to_bucket(filename=fname, ext_filename=fname)


    # visualise
    fname='{}/avg_all_stations.png'.format(options.vis_path)
    viz.plot_delay(all_times,
                   avg_delay,
                   avg_pred_delay,
                   'Average delay for all station',
                   fname)
    io._upload_to_bucket(filename=fname, ext_filename=fname)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, default=None, help='Configuration file name')
    parser.add_argument('--config_name', type=str, default=None, help='Configuration file name')
    parser.add_argument('--starttime', type=str, default='2011-02-01', help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, default='2017-03-01', help='End time of the classification data interval')
    parser.add_argument('--model_path', type=str, default=None, help='Path of TensorFlow Estimator file')
    parser.add_argument('--model_file', type=str, default=None, help='Path and filename of SciKit model file. If this is given, model_path is ignored.')
    parser.add_argument('--stations', type=str, default=None, help='List of train stations separated by comma')
    parser.add_argument('--stations_file', type=str, default='cnf/stations.json', help='Stations file, list of stations to process')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')
    parser.add_argument('--output_path', type=str, default=None, help='Path where visualizations are saved')

    options = parser.parse_args()

    _config.read(options)

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
