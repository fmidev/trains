import sys, os, argparse, logging, json, copy
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
#from tensorflow.contrib import predictor

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from lib.io import IO, ModelError
from lib.viz import Viz
from lib.bqhandler import BQHandler
from lib.modelloader import ModelLoader

from lib import config as _config
from lib.predictor import Predictor, PredictionError


STATION_SPECIFIC_CLASSIFIER = True
STATION_SPECIFIC_REGRESSOR = False

def main():
    """
    Get data from db and save it as csv
    """

    bq = BQHandler()
    io = IO(gs_bucket=options.gs_bucket)
    viz = Viz(io=io)
    predictor = Predictor(io, ModelLoader(io), options, STATION_SPECIFIC_CLASSIFIER, STATION_SPECIFIC_REGRESSOR)
    predictor.regressor_save_file = options.save_path+'/classifier.pkl'
    predictor.classifier_save_file = options.save_path+'/regressor.pkl'

    # Mean delay over the whole dataset (both train and validation),
    # used to calculate Brier Skill
    if options.y_avg:
        mean_delay = 3.375953418071136
    else:
        mean_delay = 6.011229358531166

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    # Get params
    all_param_names = list(set(options.label_params + options.feature_params + options.meta_params + options.classifier_feature_params + options.regressor_feature_params))

    # Param list is modified after retrieving data
    classifier_feature_params = copy.deepcopy(options.classifier_feature_params)
    regressor_feature_params = copy.deepcopy(options.regressor_feature_params)

    all_feature_params = list(set(options.feature_params + options.meta_params + options.classifier_feature_params + options.regressor_feature_params))
    aggs = io.get_aggs_from_param_names(all_feature_params)

    # Init error dicts
    avg_delay = {}
    avg_pred_delay = {}
    avg_proba = {}
    station_count = 0
    all_times = set()

    station_rmse = {}
    station_mae = {}
    station_r2 = {}
    station_skill = {}

    # For aggregated binary classification metrics
    time_list, target_list, y_pred_bin_list, y_pred_bin_proba_list = [], [], [], []

    # If stations are given as argument use them, else use all stations
    stationList = io.get_train_stations(options.stations_file)
    all_data = None

    if options.locations is not None:
        stations = options.locations
    else:
        stations = stationList.keys()

    # Go through stations
    for station in stations:
        stationName = '{} ({})'.format(stationList[station]['name'], station)
        logging.info('Processing station {}'.format(stationName))

        if hasattr(options, 'classifier_model_file'):
            predictor.classifier_save_file = options.classifier_model_file.replace('{location}', station)
        elif STATION_SPECIFIC_CLASSIFIER:
            predictor.classifier_save_file = options.save_path+'/{}'.format(station)+'/classifier.pkl'

        if hasattr(options, 'regressor_model_file'):
            predictor.regressor_save_file = options.regressor_model_file.replace('{location}', station)
        elif STATION_SPECIFIC_REGRESSOR:
            predictor.regressor_save_file = options.save_path+'/{}'.format(station)+'/regressor.pkl'

        station_rmse[station] = {}
        station_mae[station] = {}
        station_r2[station] = {}
        station_skill[station] = {}

        # Read data and filter desired train types (ic and commuter)
        data = bq.get_rows(starttime,
                           endtime,
                           loc_col='trainstation',
                           project=options.project,
                           dataset='trains_data',
                           table='features_testset',
                           parameters=all_param_names,
                           only_winters=options.only_winters,
                           locations=[station])

        data = io.filter_train_type(labels_df=data,
                                    train_types=['K','L'],
                                    sum_types=True,
                                    train_type_column='train_type',
                                    location_column='trainstation',
                                    time_column='time',
                                    sum_columns=['train_count', 'delay'],
                                    aggs=aggs)

        if len(data) == 0:
            continue

        if options.y_avg_hours is not None:
            data = io.calc_running_delay_avg(data, options.y_avg_hours)

        if options.y_avg:
            data = io.calc_delay_avg(data)

        if options.month:
            logging.info('Adding month to the dataset...')
            data = data.assign(month=lambda df: df.loc[:, 'time'].map(lambda x: x.month))
            if 'month' not in options.feature_params:
                options.feature_params.append('month')
            if 'month' not in options.regressor_feature_params:
                options.regressor_feature_params.append('month')
            if 'month' not in options.classifier_feature_params:
                options.classifier_feature_params.append('month')


        data.sort_values(by=['time'], inplace=True)
        logging.info('Processing {} rows...'.format(len(data)))

        if all_data is None:
            all_data = data
        else:
            all_data.append(data, ignore_index=True)

        # Pick times for creating error time series
        times = data.loc[:,'time']
        station_count += 1

        # Run prediction
        try:
            #target, y_pred = predictor.pred(times, data)
            y_pred, y_pred_bin, y_pred_bin_proba = predictor.pred(times, data)
            # Drop first times which LSTM are not able to predict
            #times = times[(len(data)-len(y_pred)):]
        except (PredictionError, ModelError) as e:
            logging.error(e)
            continue

        target = data.loc[:, options.label_params].reset_index(drop=True).values.ravel()

        if len(y_pred) < 1 or len(target) < 1:
            continue

        # Create timeseries of predicted and happended delay
        i = 0
        for t in times:
            try:
                if t not in avg_delay.keys():
                    avg_delay[t] = [target[i]]
                    avg_pred_delay[t] = [y_pred[i]]
                    if predictor.y_pred_bin_proba is not None:
                        avg_proba[t] = [predictor.y_pred_bin_proba[i, 1]]
                else:
                    avg_delay[t].append(target[i])
                    avg_pred_delay[t].append(y_pred[i])
                    if predictor.y_pred_bin_proba is not None:
                        avg_proba[t].append(predictor.y_pred_bin_proba[i, 1])
            except IndexError as e:
                # LSTM don't have first time steps because it don't
                # have necessary history
                pass
            i += 1

        # For creating visualisation
        all_times = all_times.union(set(times))

        # If only average plots are asked, continue to next station
        if options.only_avg == 1:
            continue

        # Calculate errors for given station, first for all periods and then for whole time range
        if predictor.y_pred_bin is not None:
            time_list += list(times)

            #feature_list += list()
            target_list += list(target)
            y_pred_bin_list += list(predictor.y_pred_bin)
            y_pred_bin_proba_list += list(predictor.y_pred_bin_proba)

            splits = viz._split_to_parts(list(times), [target, y_pred, predictor.y_pred_bin, predictor.y_pred_bin_proba], 2592000)
        else:
            splits = viz._split_to_parts(list(times), [target, y_pred], 2592000)

        for i in range(0, len(splits)):

            logging.info('Month {}:'.format(i+1))

            if predictor.y_pred_bin is not None:
                times_, target_, y_pred_, y_pred_bin_, y_pred_bin_proba_ = splits[i]
                viz.classification_perf_metrics(y_pred_bin_proba_, y_pred_bin_, target_, options, times_, station)
            else:
                times_, target_, y_pred_ = splits[i]

            rmse = math.sqrt(metrics.mean_squared_error(target_, y_pred_))
            mae = metrics.mean_absolute_error(target_, y_pred_)
            r2 = metrics.r2_score(target_, y_pred_)
            rmse_stat = math.sqrt(metrics.mean_squared_error(target_, np.full_like(target_, mean_delay)))
            skill = 1 - rmse / rmse_stat

            # Put errors to timeseries
            station_rmse[station][i] = rmse
            station_mae[station][i] = mae
            station_r2[station][i] = r2
            station_skill[station][i] = skill

            logging.info('RMSE of station {} month {}: {:.4f}'.format(stationName, i+1, rmse))
            logging.info('MAE of station {} month {}: {:.4f}'.format(stationName, i+1, mae))
            logging.info('R2 score of station {} month {}: {:.4f}'.format(stationName, i+1, r2))
            logging.info('Skill (RMSE) of station {} month {}: {:.4f}'.format(stationName, i+1, skill))

        mse = math.sqrt(metrics.mean_squared_error(target, y_pred))
        mae = metrics.mean_absolute_error(target, y_pred)
        r2 = metrics.r2_score(target, y_pred)
        rmse_stat = math.sqrt(metrics.mean_squared_error(target, np.full_like(target, mean_delay)))
        skill = 1 - rmse / rmse_stat

        station_rmse[station]['all'] = rmse
        station_mae[station]['all'] = mae
        station_r2[station]['all'] = r2
        station_skill[station]['all'] = skill

        logging.info('All periods:')
        logging.info('RMSE of station {} month {}: {:.4f}'.format(stationName, i+1, rmse))
        logging.info('MAE of station {} month {}: {:.4f}'.format(stationName, i+1, mae))
        logging.info('R2 score of station {} month {}: {:.4f}'.format(stationName, i+1, r2))
        logging.info('Skill (RMSE) of station {} month {}: {:.4f}'.format(stationName, i+1, skill))

        # Create csv and upload it to pucket
        times_formatted = [t.strftime('%Y-%m-%dT%H:%M:%S') for t in times]
        delay_data = {'times': times_formatted, 'delay': target, 'predicted delay': y_pred}
        fname = '{}/delays_{}.csv'.format(options.vis_path, station)
        io.write_csv(delay_data, fname, fname)

        # Draw visualisation
        fname='{}/timeseries_{}'.format(options.vis_path, station)

        if predictor.y_pred_bin_proba is not None:
            proba = predictor.y_pred_bin_proba[:,1]
            viz.plot_delay(times, target, None, 'Delay for station {}'.format(stationName), fname, all_proba=proba, proba_mode='same', color_threshold=options.class_limit)
        else:
            viz.plot_delay(times, target, y_pred, 'Delay for station {}'.format(stationName), fname, all_proba=None)

        fname='{}/scatter_all_stations.png'.format(options.vis_path)
        viz.scatter_predictions(times, target, y_pred, savepath=options.vis_path, filename='scatter_{}'.format(station))


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
    for t,l in avg_delay.items():
        avg_delay[t] = sum(l)/len(l)
    for t, l in avg_pred_delay.items():
        avg_pred_delay[t] = sum(l)/len(l)
    for t, l in avg_proba.items():
        avg_proba[t] = sum(l)/len(l)

    avg_delay = list(OrderedDict(sorted(avg_delay.items(), key=lambda t: t[0])).values())
    avg_pred_delay = list(OrderedDict(sorted(avg_pred_delay.items(), key=lambda t: t[0])).values())
    avg_proba = list(OrderedDict(sorted(avg_proba.items(), key=lambda t: t[0])).values())

    # Calculate average over all times and stations, first for all months separately, then for whole time range
    splits = viz._split_to_parts(list(times), [avg_delay, avg_pred_delay], 2592000)

    for i in range(0, len(splits)):
        times_, avg_delay_, avg_pred_delay_ = splits[i]

        try:
            rmse = math.sqrt(metrics.mean_squared_error(avg_delay_, avg_pred_delay_))
            mae = metrics.mean_absolute_error(avg_delay_, avg_pred_delay_)
            r2 = metrics.r2_score(avg_delay_, avg_pred_delay_)
            rmse_stat = math.sqrt(metrics.mean_squared_error(avg_delay_, np.full_like(avg_delay_, mean_delay)))
            skill = 1 - rmse/rmse_stat
        except ValueError:
            logging.warning('Zero samples in some class')
            continue

        logging.info('Month: {}'.format(i+1))
        logging.info('RMSE of average delay over all stations: {:.4f}'.format(rmse))
        logging.info('MAE of average delay over all stations: {:.4f}'.format(mae))
        logging.info('R2 score of average delay over all stations: {:.4f}'.format(r2))
        logging.info('Skill score (RMSE) of average delay over all stations: {:.4f}'.format(skill))

        # Write average data into file
        avg_errors = {'rmse': rmse, 'mae': mae, 'r2': r2,
                      'skill': skill,
                      'nro_of_samples': len(avg_delay)}
        fname = '{}/avg_erros_{}.csv'.format(options.vis_path, i)
        io.dict_to_csv(avg_errors, fname, fname)


    rmse = math.sqrt(metrics.mean_squared_error(avg_delay, avg_pred_delay))
    #rmse_mean = np.mean(list(station_rmse.values()))
    mae = metrics.mean_absolute_error(avg_delay, avg_pred_delay)
    #mae_mean = np.mean(list(station_mae.values()))
    r2 = metrics.r2_score(avg_delay, avg_pred_delay)
    rmse_stat = math.sqrt(metrics.mean_squared_error(avg_delay, np.full_like(avg_delay, mean_delay)))
    skill = 1 - rmse/rmse_stat
    #skill_mean = 1 - rmse_mean/rmse_stat

    logging.info('All periods:')
    logging.info('RMSE of average delay over all stations: {:.4f}'.format(rmse))
    #logging.info('Average RMSE of all station RMSEs: {:.4f}'.format(rmse_mean))
    logging.info('MAE of average delay over all stations: {:.4f}'.format(mae))
    #logging.info('Average MAE of all station MAEs: {:.4f}'.format(mae_mean))
    logging.info('R2 score of average delay over all stations: {:.4f}'.format(r2))
    logging.info('Skill score (RMSE) of average delay over all stations: {:.4f}'.format(skill))
    #logging.info('Skill score (avg RMSE) of all stations: {:.4f}'.format(skill_mean))

    # Write average data into file
    avg_errors = {'rmse': rmse, 'mae': mae, 'r2': r2,
                  #'rmse_mean': rmse_mean,
                  #'mae_mean': mae_mean,
                  'skill': skill,
                  #'skill_mean': skill_mean,
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
    if not avg_proba:
        proba = None
    else:
        proba = avg_proba
    fname='{}/timeseries_avg_all_stations.png'.format(options.vis_path)

    if predictor.y_pred_bin is not None:
        viz.plot_delay(all_times, avg_delay, None, 'Average delay for all station', fname, all_proba=proba, proba_mode='same', color_threshold=options.class_limit)
    else:
        viz.plot_delay(all_times, avg_delay, avg_pred_delay, 'Average delay for all station', fname)


    fname='{}/scatter_all_stations.png'.format(options.vis_path)
    viz.scatter_predictions(all_times, avg_delay, avg_pred_delay, savepath=options.vis_path, filename='scatter_all_stations')

    # Binary classification metrics
    if predictor.y_pred_bin is not None:
        all_data.sort_values(by=['time'], inplace=True)
        times = all_data.loc[:,'time'].values
        try:
            target, y_pred = predictor.pred(times, all_data)
            # Drop first times which LSTM are not able to predict
            times = times[(len(all_data)-len(y_pred)):]
            splits = viz._split_to_parts(list(times), [target, y_pred, predictor.y_pred_bin, predictor.y_pred_bin_proba], 2592000)

            for i in range(0, len(splits)):
                #times_, target_, y_pred_bin_, y_pred_bin_proba_ = splits[i]
                times_, target_, y_pred_, y_pred_bin_, y_pred_bin_proba_ = splits[i]
                viz.classification_perf_metrics(y_pred_bin_proba_, y_pred_bin_, target_, options, times_, 'all')
        except (PredictionError, ModelError) as e:
            logging.error(e)
            pass


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

    logging.info('Using configuration {} | {}'.format(options.config_filename, options.config_name))

    main()
