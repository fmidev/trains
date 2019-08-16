import sys, os, math
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
from configparser import ConfigParser

import numpy as np
import pandas as pd

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ARDRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


from lib.io import IO
from lib.viz import Viz
from lib.bqhandler import BQHandler
from lib import imputer
from lib import config as _config

def main():
    """
    Get data from db and save it as csv
    """

    bq = BQHandler()
    io = IO(gs_bucket=options.gs_bucket)
    viz = Viz(io=io)

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    if options.model == 'rf':
        model = RandomForestRegressor(n_estimators=options.n_estimators,
                                      n_jobs=-1,
                                      min_samples_leaf=options.min_samples_leaf,
                                      min_samples_split=options.min_samples_split,
                                      max_features=options.max_features,
                                      max_depth=options.max_depth,
                                      bootstrap=options.bootstrap
                                      )
    elif options.model == 'lr':
        model = SGDRegressor(warm_start=True,
                             max_iter=options.n_loops,
                             shuffle=options.shuffle,
                             power_t=options.power_t,
                             penalty=options.regularizer,
                             learning_rate=options.learning_rate,
                             eta0=options.eta0,
                             alpha=options.alpha,
                             tol=0.0001
                             )
    elif options.model == 'svr':
        model = SVR()
    elif options.model == 'ard':
        model = ARDRegression(n_iter=options.n_loops,
                              alpha_1=options.alpha_1,
                              alpha_2=options.alpha_2,
                              lambda_1=options.lambda_1,
                              lambda_2=options.lambda_2,
                              threshold_lambda=options.threshold_lambda,
                              fit_intercept=options.fit_intercept,
                              copy_X=options.copy_X)
    elif options.model == 'gp':
        k_long_term = 66.0**2 * RBF(length_scale=67.0)
        k_seasonal = 2.4**2 * RBF(length_scale=90.0)* ExpSineSquared(length_scale=150, periodicity=1.0, periodicity_bounds=(0,10000))
        k_medium_term = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
        k_noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)
        #kernel_gpml = k_long_term + k_seasonal + k_medium_term + k_noise
        kernel_gpml = k_long_term + k_seasonal + k_medium_term + k_noise

        model = GaussianProcessRegressor(kernel=kernel_gpml, #alpha=0,
                                         optimizer=None, normalize_y=True)

    if options.pca:
        ipca = IncrementalPCA(n_components=options.pca_components,
                              whiten = options.whiten,
                              copy = False)

    rmses, maes, r2s, skills, start_times, end_times, end_times_obj = [], [], [], [], [], [], []
    X_complete = [] # Used for feature selection

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
                               parameters=all_param_names,
                               only_winters=options.only_winters)
            data = io.filter_train_type(labels_df=data,
                                        train_types=options.train_types,
                                        sum_types=True,
                                        train_type_column='train_type',
                                        location_column='trainstation',
                                        time_column='time',
                                        sum_columns=['train_count', 'delay'],
                                        aggs=aggs)

            if options.y_avg_hours is not None:
                data = io.calc_running_delay_avg(data, options.y_avg_hours)

            if options.y_avg:
                data = io.calc_delay_avg(data)

            data.sort_values(by=['time', 'trainstation'], inplace=True)

            if options.impute:
                logging.info('Imputing missing values...')
                data.drop(columns=['train_type'], inplace=True)
                data = imputer.fit_transform(data)
                data.loc[:, 'train_type'] = None

            if options.month:
                logging.info('Adding month to the dataset...')
                data['month'] = data['time'].map(lambda x: x.month)
                if 'month' not in options.feature_params:
                    options.feature_params.append('month')

            if options.model == 'ard' and len(data) > options.n_samples:
                logging.info('Sampling {} values from data...'.format(options.n_samples))
                data = data.sample(options.n_samples)

            l_data = data.loc[:, options.label_params]
            f_data = data.loc[:, options.feature_params]

        except ValueError as e:
            f_data, l_data = [], []

        if len(f_data) < 2 or len(l_data) < 2:
            start = end
            end = start + timedelta(days=int(options.day_step), hours=int(options.hour_step))
            continue

        logging.info('Processing {} rows...'.format(len(f_data)))

        target = l_data.astype(np.float32).values.ravel()
        features = f_data.astype(np.float32).values

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1)

        logging.debug('Features shape: {}'.format(X_train.shape))

        if options.normalize:
            logging.info('Normalizing data...')
            xscaler, yscaler = StandardScaler(), StandardScaler()

            X_train = xscaler.fit_transform(X_train)
            X_test = xscaler.transform(X_test)

            y_train = yscaler.fit_transform(y_train)
            y_test = yscaler.transform(y_test)

        if options.pca:
            logging.info('Doing PCA analyzis for the data...')
            X_train = ipca.fit_transform(X_train)
            fname = options.output_path+'/ipca_explained_variance.png'
            viz.explained_variance(ipca, fname)
            #io._upload_to_bucket(filename=fname, ext_filename=fname)
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
            io.report_cv_results(random_search.cv_results_, fname)
            #io._upload_to_bucket(filename=fname, ext_filename=fname)
            sys.exit()
        else:
            logging.info('Training...')
            if options.model in ['rf', 'svr', 'ard', 'gp']:
                model.fit(X_train, y_train)
                if options.feature_selection:
                    X_complete = X_train
                    y_complete = y_train
                    meta_complete = data.loc[:,options.meta_params]
            else:
                model.partial_fit(X_train, y_train)
                if options.feature_selection:
                    try:
                        X_complete = np.append(X_complete, X_train)
                        y_complete = np.append(Y_complete, y_train)
                        meta_complete = meta_complete.append(data.loc[:, options.meta_params])
                    except (ValueError, NameError):
                        X_complete = X_train
                        y_complete = y_train
                        meta_complete = data.loc[:, options.meta_params]

        # Metrics
        # Mean delay over the whole dataset (both train and validation),
        # used to calculate Brier Skill
        mean_delay = 6.011229358531166
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse_stat = math.sqrt(mean_squared_error(y_test, np.full_like(y_test, mean_delay)))
        skill = 1 - rmse / rmse_stat

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        skills.append(skill)
        start_times.append(start.strftime('%Y-%m-%dT%H:%M:%S'))
        end_times.append(end.strftime('%Y-%m-%dT%H:%M:%S'))
        end_times_obj.append(end)

        if options.model in ['rf', 'lr', 'ard', 'gp']:
            logging.info('R2 score for training: {}'.format(model.score(X_train, y_train)))

        logging.info('RMSE: {}'.format(rmse))
        logging.info('MAE: {}'.format(mae))
        logging.info('R2 score: {}'.format(r2))
        logging.info('Brier Skill Score score: {}'.format(skill))

        start = end
        end = start + timedelta(days=int(options.day_step), hours=int(options.hour_step))
        if end > endtime: end = endtime

    io.save_scikit_model(model, filename=options.save_file, ext_filename=options.save_file)
    if options.normalize:
        fname = options.save_path+'/xscaler.pkl'
        io.save_scikit_model(xscaler, filename=fname, ext_filename=fname)
        fname = options.save_path+'/yscaler.pkl'
        io.save_scikit_model(yscaler, filename=fname, ext_filename=fname)

    if options.model == 'rf':
        fname = options.output_path+'/rfc_feature_importance.png'
        viz.rfc_feature_importance(model.feature_importances_, fname, feature_names=options.feature_params)
        #io._upload_to_bucket(filename=fname, ext_filename=fname)

    try:
        fname = options.output_path+'/learning_over_time.png'
        viz.plot_learning_over_time(end_times_obj, rmses, maes, r2s, filename=fname)
        #io._upload_to_bucket(filename=fname, ext_filename=fname)
    except Exception as e:
        logging.error(e)

    error_data = {'start_times': start_times,
                  'end_times': end_times,
                  'rmse': rmses,
                  'mae': maes,
                  'r2': r2s,
                  'skill': skills}
    fname = '{}/training_time_validation_errors.csv'.format(options.output_path)
    io.write_csv(error_data, filename=fname, ext_filename=fname)

    # Try doing feature selection
    if options.feature_selection:
        logging.info('Doing feature selection...')
        selector = SelectFromModel(model, prefit=True)
        print(pd.DataFrame(data=X_complete))
        X_selected = selector.transform(X_complete)

        selected_columns = f_data.columns.values[selector.get_support()]
        logging.info('Selected following parameters: {}'.format(selected_columns))
        data_sel = meta_complete.join(pd.DataFrame(data=y_complete, columns=options.label_params)).join(pd.DataFrame(data=X_selected, columns=selected_columns))

        print(pd.DataFrame(data=X_selected, columns=selected_columns))
        print(data_sel)


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


    logging.info('Using configuration {}|{}'.format(options.config_filename,
                                                    options.config_name))

    main()
