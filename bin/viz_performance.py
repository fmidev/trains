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
from lib import modelloader as _ml
from lib import predictor as _predictor

def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz()
    model_loader = _ml.ModelLoader(io)
    predictor = _predictor.Predictor(io, model_loader, options)

    # Mean delay over the whole dataset (both train and validation),
    # used to calculate Brier Skill
    mean_delay = 6.011229358531166

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    # Get params
    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    # Init error dicts
    avg_delay = {}
    avg_pred_delay = {}
    station_count = 0
    all_times = set()

    station_rmse = {}
    station_mae = {}
    station_r2 = {}
    station_skill = {}

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

        # Run prediction
        target, y_pred = predictor.pred(times, data)

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

        # If only average plots are asked, continue to next station
        if options.only_avg == 1:
            continue

        # Calculate errors for given station
        rmse = math.sqrt(metrics.mean_squared_error(target, y_pred))
        mae = metrics.mean_absolute_error(target, y_pred)
        r2 = metrics.r2_score(target, y_pred)
        rmse_stat = math.sqrt(metrics.mean_squared_error(target, np.full_like(target, mean_delay)))
        skill = 1 - rmse / rmse_stat

        # Put errors to timeseries
        station_rmse[station] = rmse
        station_mae[station] = mae
        station_r2[station] = r2
        station_skill[station] = skill

        logging.info('RMSE of station {}: {:.4f}'.format(stationName, rmse))
        logging.info('MAE of station {}: {:.4f}'.format(stationName, mae))
        logging.info('R2 score of station {}: {:.4f}'.format(stationName, r2))
        logging.info('Skill (RMSE) of station {}: {:.4f}'.format(stationName, skill))

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
    fname = '{}/station_mae.csv'.format(options.vis_path)
    io.dict_to_csv(station_mae, fname, fname)
    fname = '{}/station_r2.csv'.format(options.vis_path)
    io.dict_to_csv(station_r2, fname, fname)
    fname = '{}/station_skill_rmse.csv'.format(options.vis_path)
    io.dict_to_csv(station_skill, fname, fname)

    # Create timeseries of avg actual delay and predicted delay
    all_times = sorted(list(all_times))
    # print(avg_pred_delay)
    #print(avg_pred_delay)
    for t,l in avg_delay.items():
        avg_delay[t] = sum(l)/len(l)
    for t, l in avg_pred_delay.items():
        avg_pred_delay[t] = sum(l)/len(l)
    avg_delay = list(OrderedDict(sorted(avg_delay.items(), key=lambda t: t[0])).values())
    avg_pred_delay = list(OrderedDict(sorted(avg_pred_delay.items(), key=lambda t: t[0])).values())

    # Calculate average over all times and stations
    rmse = math.sqrt(metrics.mean_squared_error(avg_delay, avg_pred_delay))
    rmse_mean = np.mean(list(station_rmse.values()))
    mae = metrics.mean_absolute_error(avg_delay, avg_pred_delay)
    mae_mean = np.mean(list(station_mae.values()))
    r2 = metrics.r2_score(avg_delay, avg_pred_delay)
    rmse_stat = math.sqrt(metrics.mean_squared_error(avg_delay, np.full_like(avg_delay, mean_delay)))
    skill = 1 - rmse/rmse_stat
    skill_mean = 1 - rmse_mean/rmse_stat

    logging.info('RMSE of average delay over all stations: {:.4f}'.format(rmse))
    logging.info('Average RMSE of all station RMSEs: {:.4f}'.format(rmse_mean))
    logging.info('MAE of average delay over all stations: {:.4f}'.format(mae))
    logging.info('Average MAE of all station MAEs: {:.4f}'.format(mae_mean))
    logging.info('R2 score of average delay over all stations: {:.4f}'.format(r2))
    logging.info('Skill score (RMSE) of average delay over all stations: {:.4f}'.format(skill))
    logging.info('Skill score (avg RMSE) of all stations: {:.4f}'.format(skill_mean))

    # Write average data into file
    avg_errors = {'rmse': rmse, 'mae': mae, 'r2': r2,
                  'rmse_mean': rmse_mean, 'mae_mean': mae_mean,
                  'skill': skill, 'skill_mean': skill_mean,
                  'nro_of_samples': len(avg_delay)}
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
    parser.add_argument('--only_avg', type=int, default=0, help='Create only avg plot')
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
