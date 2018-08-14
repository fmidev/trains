import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np
from configparser import ConfigParser

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq

def report_cv_results(results, filename=None, n_top=3):
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

def _config(options): #config_filename, section):

    def _path(name, root_dir):
        ''' Read path from options and create it if not exists'''
        val = getattr(options, name, None)
        if val is None or val == 'None':
            val = root_dir+'/'+options.model+'/'+options.feature_dataset+'/'+options.config_name

        if not os.path.exists(val):
            os.makedirs(val)

        setattr(options, name, val)

    def _fval(name):
        ''' Convert float val to float taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = float(val)
        else:
            val = None
        setattr(options, name, val)

    def _bval(name):
        ''' Convert option from int to bool'''
        val = getattr(options, name, False)
        if int(val) == 1: val = True
        else: val = False
        setattr(options, name, val)

    def _intval(name):
        ''' Convert int val to integer taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = int(val)
        else:
            val = None
        setattr(options, name, val)

    parser = ConfigParser()
    parser.read(options.config_filename)

    if parser.has_section(options.config_name):
        params = parser.items(options.config_name)
        for param in params:
            setattr(options, param[0], param[1])

        options.feature_params = options.feature_params.split(',')
        options.label_params = options.label_params.split(',')
        options.meta_params = options.meta_params.split(',')

        if options.dev == 1: options.n_loops = 100

        _path('save_path', 'models')
        _path('output_path', 'results')
        _path('log_dir', '/tmp')
        options.save_file = options.save_path+'/model.pkl'

        _bval('cv')
        _bval('pca')
        _bval('whiten')
        _bval('normalize')
        _bval('impute')
        _bval('shuffle')

        _fval('alpha')
        _fval('eta0')
        _fval('power_t')

        _intval('pca_components')
        _intval('n_loops')

        return options
    else:
        raise Exception('Section {} not found in the {} file'.format(options.config_name, options.config_filename))

    return tables

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

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    if options.model == 'rf':
        model = RandomForestRegressor(n_estimators=50,
                                      warm_start=True,
                                      n_jobs=-1)
    elif options.model == 'lr':
        model = SGDRegressor(warm_start=True,
                             max_iter=options.n_loops,
                             shuffle=options.shuffle,
                             power_t=options.power_t,
                             penalty=options.penalty,
                             learning_rate=options.learning_rate,
                             eta0=options.eta0,
                             alpha=options.alpha)
    elif options.model == 'svr':
        model = SVR()

    if options.pca:
        ipca = IncrementalPCA(n_components=options.pca_components,
                              whiten = options.whiten,
                              copy = False)

    rmses, maes, r2s, start_times, end_times, end_times_obj = [], [], [], [], [], []

    start = starttime
    end = start + timedelta(days=int(options.day_step), hours=int(options.hour_step))
    if end > endtime: end = endtime

    while end <= endtime and start < end:
        logging.info('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'),
                                                            end.strftime('%Y-%m-%d %H:%M')))

        try:
            logging.info('Reading data...')
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

            l_data = data.loc[:,options.meta_params + options.label_params]
            f_data = data.loc[:,options.meta_params + options.feature_params]

            print(f_data.columns)
            print(options.feature_params)

        except ValueError as e:
            f_data, l_data = [], []

        if len(f_data) == 0 or len(l_data) == 0:
            start = end
            end = start + timedelta(days=int(options.day_step), hours=int(options.hour_step))
            continue

        f_data.rename(columns={'trainstation':'loc_name'}, inplace=True)

        logging.debug('Labels shape: {}'.format(l_data.shape))

        logging.info('Processing {} rows...'.format(len(f_data)))

        assert l_data.shape[0] == f_data.shape[0]

        target = l_data['delay'].astype(np.float32).values
        features = f_data.drop(columns=['loc_name', 'time']).astype(np.float32).values

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33)

        logging.debug('Features shape: {}'.format(X_train.shape))

        n_samples, n_dims = X_train.shape

        if options.impute:
            logging.info('Imputing missing values with {}...'.format(options.impute_strategy))
            imputer = Imputer(missing_values=-99, strategy=options.impute_strategy)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.fit_transform(X_test)

        if options.normalize:
            logging.info('Normalizing data...')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

        if options.pca:
            logging.info('Doing PCA analyzis for the data...')
            X_train = ipca.fit_transform(X_train)
            fname = options.output_path+'/ipca_explained_variance.png'
            viz.explained_variance(ipca, fname)
            io._upload_to_bucket(filename=fname, ext_filename=fname)
            X_test = ipca.fit_transform(X_test)

        logging.debug('Features shape after pre-processing: {}'.format(X_train.shape))

        if options.cv:
            logging.info('Doing random search for hyper parameters...')

            if options.model == 'rf':
                param_grid = {"n_estimators": [10, 100, 200, 800],
                              "max_depth": [3, 20, None],
                              "max_features": ["auto", "sqrt", "log2", None],
                              "min_samples_split": [2,5,10],
                              "min_samples_leaf": [1, 2, 4, 10],
                              "bootstrap": [True, False]}
            elif options.model == 'lr':
                param_grid = {"penalty": [None, 'l2', 'l1'],
                              "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
                              "l1_ratio": [0.1, 0.15, 0.2, 0.5],
                              "shuffle": [True, False],
                              "learning_rate": ['constant', 'optimal', 'invscaling'],
                              "eta0": [0.001, 0.01, 0.1],
                              "power_t": [0.1, 0.25, 0.5]}
            elif options.model == 'svr':
                param_grid = {"C": [0.001, 0.01, 0.1, 1, 10],
                              "epsilon": [0.01, 0.1, 0.5],
                              "kernel": ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'],
                              "degree": [2, 3 ,4],
                              "shrinking": [True, False],
                              "gamma": [0.001, 0.01, 0.1],
                              "coef0": [0, 0.1, 1]}
            else:
                raise("No param_grid set for given model ({})".format(options.model))

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
            logging.info('Training...')
            if options.model in ['rf', 'svr']:
                model.fit(X_train, y_train)
            else:
                model.partial_fit(X_train, y_train)

        # Metrics
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        start_times.append(start.strftime('%Y-%m-%dT%H:%M:%S'))
        end_times.append(end.strftime('%Y-%m-%dT%H:%M:%S'))
        end_times_obj.append(end)

        if options.model in ['rf', 'lr']:
            logging.info('R2 score for training: {}'.format(model.score(X_train, y_train)))

        logging.info('RMSE: {}'.format(rmse))
        logging.info('MAE: {}'.format(mae))
        logging.info('R2 score: {}'.format(r2))

        start = end
        end = start + timedelta(days=int(options.day_step), hours=int(options.hour_step))
        if end > endtime: end = endtime

    io.save_scikit_model(model, filename=options.save_file, ext_filename=options.save_file)
    if options.model == 'rf':
        fname = options.output_path+'/rfc_feature_importance.png'
        viz.rfc_feature_importance(model.feature_importances_, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)

    try:
        fname = options.output_path+'/learning_over_time.png'
        viz.plot_learning_over_time(end_times_obj, rmses, maes, r2s, filename=fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)
    except Exception as e:
        logging.error(e)

    error_data = {'start_times': start_times,
                  'end_times': end_times,
                  'rmse': rmses,
                  'mae': maes,
                  'r2': r2s}
    fname = '{}/errors.csv'.format(options.output_path)
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

    _config(options)

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
