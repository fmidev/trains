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

    # Load model
    if options.tf:
        io._download_dir_from_bucket(options.save_path, options.save_path, force=True)
        sess = tf.Session()
        saver = tf.train.import_meta_graph(options.save_file+'.meta')
        saver.restore(sess, tf.train.latest_checkpoint(options.save_path))
        graph = tf.get_default_graph()

        # for i in tf.get_default_graph().get_operations():
        #     print(i.name)
        # sys.exit()
        X = graph.get_tensor_by_name("inputs/X:0")
        op_y_pred = graph.get_tensor_by_name("out_hidden/y_pred/y_pred:0")
    else:
        io._download_from_bucket(options.save_file, options.save_file)
        predictor = io.load_scikit_model(options.save_file)

#        print(op_y_pred)
#    sys.exit()

    # Init error dicts
    avg_delay = {}
    avg_pred_delay = {}
    station_count = 0
    all_times = set()

    station_rmse = {}
    station_median_abs_err = {}
    station_r2 = {}

    # If stations are given as argument use them, else use all stations
    stationList = io.get_train_stations(options.stations_file)

    if options.stations is not None:
        stations = options.stations.split(',')
    else:
        stations = stationList.keys()

    # Go through stations
    for station in stations:
        stationName = '{} ({})'.format(stationList[station]['name'], station)
        logging.info('Processing station {}'.format(stationName))

        # Read data and filter desired train types (ic and commuter)
        data = bq.get_rows(starttime,
                           endtime,
                           loc_col='trainstation',
                           project=options.project,
                           dataset='trains_testset',
                           table='features_1',
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

        if len(data) == 0:
            continue

        if options.y_avg_hours is not None:
            data = io.calc_running_delay_avg(data, options.y_avg_hours)

        data.sort_values(by=['time', 'trainstation'], inplace=True)
        logging.info('Processing {} rows...'.format(len(data)))

        # Pick times for creating error time series
        times = data.loc[:,'time']
        station_count += 1

        if not options.tf:
            # Pick feature and label data from all data
            l_data = data.loc[:,options.meta_params + options.label_params]
            f_data = data.loc[:,options.meta_params + options.feature_params]

            assert l_data.shape[0] == f_data.shape[0]

            target = l_data['delay'].astype(np.float64).values.ravel()
            features = f_data.drop(columns=['trainstation', 'time']).astype(np.float64).values

        # Run prediction
        if options.tf:
            if options.normalize:
                fname=options.save_path+'/yscaler.pkl'
                io._download_from_bucket(fname, fname, force=True)
                yscaler = io.load_scikit_model(fname)

            y_pred, target = [], []
            logging.info('Predicting... ')
            start = 0
            end = options.time_steps
            first = True
            while end <= len(times):
                input_batch, target_batch = io.extract_batch(data,
                                                             options.time_steps,
                                                             batch_size=-1,
                                                             pad_strategy=options.pad_strategy,
                                                             quantile=options.quantile,
                                                             start=start,
                                                             end=end)
                if len(input_batch) < options.time_steps:
                    break

                feed_dict={X: input_batch}
                y_pred_batch = sess.run(op_y_pred, feed_dict=feed_dict).ravel()
                if options.normalize:
                    y_pred_batch = yscaler.inverse_transform(y_pred_batch)

                # print(y_pred_batch)
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

        else:
            y_pred = predictor.predict(features)

        if len(y_pred) < 1 or len(target) < 1:
            continue
            
        # Create timeseries of predicted and happended delay
        i = 0
        for t in times:
            try:
                if t not in avg_delay.keys():
                    avg_delay[t] = [target[i]]
                    avg_pred_delay[t] = [y_pred[i]]
                else:
                    avg_delay[t].append(target[i])
                    avg_pred_delay[t].append(y_pred[i])
            except IndexError as e:
                logging.error(e)
            i += 1

        # For creating visualisation
        all_times = all_times.union(set(times))

        # Calculate errors for given station
        rmse = math.sqrt(metrics.mean_squared_error(target, y_pred))
        median_abs_err = metrics.median_absolute_error(target, y_pred)
        r2 = metrics.r2_score(target, y_pred)

        # Put errors to timeseries
        station_rmse[station] = rmse
        station_median_abs_err[station] = median_abs_err
        station_r2[station] = r2

        logging.info('RMSE for station {}: {}'.format(stationName, rmse))
        logging.info('Median absolute error for station {}: {}'.format(stationName, median_abs_err))
        logging.info('R2 score for station {}: {}'.format(stationName, r2))

        # Create csv and upload it to pucket
        times_formatted = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in times]
        delay_data = {'times': times_formatted, 'delay': target, 'predicted delay': y_pred}
        fname = '{}/delays_{}.csv'.format(options.vis_path, station)
        io.write_csv(delay_data, fname, fname)

        # Draw visualisation
        fname='{}/{}.png'.format(options.vis_path, station)
        viz.plot_delay(times, target, y_pred, 'Delay for station {}'.format(stationName), fname)
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
    for t,l in avg_delay.items():
        avg_delay[t] = sum(l)/len(l)
    for t, l in avg_pred_delay.items():
        avg_pred_delay[t] = sum(l)/len(l)
    avg_delay = list(OrderedDict(sorted(avg_delay.items(), key=lambda t: t[0])).values())
    avg_pred_delay = list(OrderedDict(sorted(avg_pred_delay.items(), key=lambda t: t[0])).values())

    # Calculate average over all times and stations
    rmse = math.sqrt(metrics.mean_squared_error(avg_delay, avg_pred_delay))
    median_abs_err = metrics.median_absolute_error(avg_delay, avg_pred_delay)
    r2 = metrics.r2_score(avg_delay, avg_pred_delay)

    logging.info('RMSE for average delay over all stations: {}'.format(rmse))
    logging.info('Median absolute error for average delay over all stations: {}'.format(median_abs_err))
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

    # visualise
    fname='{}/avg_all_stations.png'.format(options.vis_path)
    viz.plot_delay(all_times, avg_delay, avg_pred_delay, 'Average delay for all station', fname)
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
