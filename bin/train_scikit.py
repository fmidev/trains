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

from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ARDRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, DotProduct, PairwiseKernel
from lib.model_functions.LocalizedLasso import LocalizedLasso
from lib.model_functions.NetworkLasso import NetworkLasso

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from lib.io import IO
from lib.viz import Viz
from lib.bqhandler import BQHandler
from lib import config as _config

def main():
    """
    Get data from db and save it as csv
    """

    bq = BQHandler()
    io = IO(gs_bucket=options.gs_bucket)
    viz = Viz(io=io)

    location = 'all'
    if options.locations is not None:
        location = options.locations[0]

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
    elif options.model == 'gbdt':        
        model = GradientBoostingRegressor(
            subsample=options.subsample,
            n_estimators=options.n_estimators,
            min_samples_split=options.min_samples_split,
            max_features=options.max_features,
            max_depth=options.max_depth,
            loss=options.loss,
            learning_rate=options.gbdt_learning_rate,
            ccp_alpha=options.ccp_alpha
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
        kernel = PairwiseKernel(metric='laplacian') *  DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, alpha=options.noise_level) #alpha correspondes to white kernel
    elif options.model == 'llasso':
        model = LocalizedLasso(num_iter=options.n_loops,
                               batch_size=options.batch_size)
    elif options.model == 'nlasso':
        model = NetworkLasso(num_iter=options.n_loops,
                             batch_size=options.batch_size)

        graph_data = pd.read_csv(options.graph_data, names=['date', 'start_hour', 'src', 'dst', 'type', 'sum_delay','sum_ahead','add_delay','add_ahead','train_count'])

        #stations_to_pick = options.stations_to_pick.split(',')
        #graph = model.fetch_connections(graph_data, stations_to_pick)
        model.fetch_connections(graph_data)

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

        # Load data ############################################################
        try:
            logging.info('Reading data...')
            data = bq.get_rows(start,
                               end,
                               loc_col='trainstation',
                               project=options.project,
                               dataset=options.feature_dataset,
                               table=options.feature_table,
                               locations=options.locations,
                               parameters=all_param_names,
                               only_winters=options.only_winters,
                               reason_code_table=options.reason_code_table,
                               reason_codes_exclude=options.reason_codes_exclude,
                               reason_codes_include=options.reason_codes_include
                               )
            data = io.filter_train_type(labels_df=data,
                                        train_types=options.train_types,
                                        sum_types=True,
                                        train_type_column='train_type',
                                        location_column='trainstation',
                                        time_column='time',
                                        sum_columns=['train_count', 'delay'],
                                        aggs=aggs)

            # Filter only timesteps with large distribution in the whole network
            if options.filter_delay_limit is not None:
                data = io.filter_delay_with_limit(data, options.filter_delay_limit)

            if options.n_samples is not None and options.n_samples < data.shape[0]:
                logging.info('Sampling {} values from data...'.format(options.n_samples))
                data = data.sample(options.n_samples)

            if options.y_avg_hours is not None:
                data = io.calc_running_delay_avg(data, options.y_avg_hours)

            if options.y_avg:
                data = io.calc_delay_avg(data)

            data.sort_values(by=['time', 'trainstation'], inplace=True)

            if options.month:
                logging.info('Adding month to the dataset...')
                data['month'] = data['time'].map(lambda x: x.month)
                if 'month' not in options.feature_params:
                    options.feature_params.append('month')

            l_data = data.loc[:, options.label_params]
            f_data = data.loc[:, options.feature_params]

        except ValueError as e:
            f_data, l_data = [], []

        if len(f_data) < 2 or len(l_data) < 2:
            start = end
            end = start + timedelta(days=int(options.day_step), hours=int(options.hour_step))
            continue

        logging.info('Processing {} rows...'.format(len(f_data)))

        train, test = train_test_split(data, test_size=0.1)
        X_train = train.loc[:, options.feature_params].astype(np.float32).values
        y_train = train.loc[:, options.label_params].astype(np.float32).values.ravel()
        X_test = test.loc[:, options.feature_params].astype(np.float32).values
        y_test = test.loc[:, options.label_params].astype(np.float32).values.ravel()

        logging.debug('Features shape: {}'.format(X_train.shape))

        if options.normalize:
            logging.info('Normalizing data...')
            xscaler, yscaler = StandardScaler(), StandardScaler()

            X_train = xscaler.fit_transform(X_train)
            X_test = xscaler.transform(X_test)

            if len(options.label_params) == 1:
                y_train = yscaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            else:
                y_train = yscaler.fit_transform(y_train)

        if options.pca:
            logging.info('Doing PCA analyzis for the data...')
            X_train = ipca.fit_transform(X_train)
            fname = options.output_path+'/ipca_explained_variance.png'
            viz.explained_variance(ipca, fname)
            #io._upload_to_bucket(filename=fname, ext_filename=fname)
            X_test = ipca.fit_transform(X_test)

        if options.model == 'llasso':
            graph_data = pd.read_csv(options.graph_data, names=['date', 'start_hour', 'src', 'dst', 'type', 'sum_delay','sum_ahead','add_delay','add_ahead','train_count'])
            graph = model.fetch_connections(graph_data)

        logging.debug('Features shape after pre-processing: {}'.format(X_train.shape))



        # FIT ##################################################################

        if options.cv:
            logging.info('Doing random search for hyper parameters...')

            if options.model == 'rf':
                param_grid = {"n_estimators": [5, 10, 50, 100],
                              "max_depth": [3, 20, None],
                              "max_features": ["auto", "sqrt", "log2", None],
                              "min_samples_split": [2,5,10],
                              "min_samples_leaf": [1, 2, 4, 10],
                              "bootstrap": [True, False]}
            elif options.model == 'gbdt':
                param_grid = {
                    "loss": ['ls', 'lad', 'huber'],
                    'learning_rate': [.0001, .001, .01, .1],
                    "n_estimators": [10, 50, 100, 200],
                    'subsample': [.1, .25, .5, 1],
                    "min_samples_split": [2,5,10],
                    "max_depth": [3, 10, 20, None],
                    "max_features": ["auto", "sqrt", "log2", None],
                    'ccp_alpha': [0, .001, 0.1]
                    }
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
            elif options.model == 'gp':
                param_grid = {'alpha': [0.1, 0.5, 1, 2, 3, 4, 5, 10]}
            else:
                raise("No param_grid set for given model ({})".format(options.model))

            random_search = RandomizedSearchCV(model,
                                               param_distributions=param_grid,
                                               n_iter=int(options.n_iter_search),
                                               scoring='neg_root_mean_squared_error',
                                               n_jobs=-1,
                                               refit=True,
                                               return_train_score=True,
                                               cv=TimeSeriesSplit(n_splits=5),
                                               verbose=1
                                               )

            random_search.fit(X_train, y_train)
            logging.info("RandomizedSearchCV done.")
            fname = options.output_path+'/random_search_cv_results.txt'
            io.report_cv_results(random_search.cv_results_, fname, fname)
            model = random_search.best_estimator_
        else:
            logging.info('Training...')
            if options.model in ['rf', 'svr', 'ard', 'gp','gbdt']:
                model.fit(X_train, y_train)
                if options.feature_selection:
                    X_complete = X_train
                    y_complete = y_train
                    meta_complete = data.loc[:,options.meta_params]
            elif options.model in ['llasso']:
                model.fit(X_train, y_train, stations=train.loc[:, 'trainstation'].values)
            elif options.model in ['nlasso']:
                model.partial_fit(X_train, y_train, stations=train.loc[:, 'trainstation'].values)
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




        # EVALUATE #############################################################

        # Mean delay over the whole dataset (both train and validation),
        # used to calculate Brier Skill
        mean_delay = options.mean_delay
        if mean_delay is None:
            mean_delay = 3.375953418071136 if options.y_avg else 6.011229358531166
        
        # Check training score to estimate amount of overfitting
        # Here we assume that we have a datetime index (from time columns)
        y_pred_train = model.predict(X_train)
        if options.normalize:
            y_pred_train = yscaler.inverse_transform(y_pred_train)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        rmse_stat_train = math.sqrt(mean_squared_error(y_train, np.full_like(y_train, mean_delay)))
        skill_train = 1 - rmse_train / rmse_stat_train
                
        logging.info('Training data metrics:\nRMSE: {}\nMAE: {}\nR2: {}\nBSS: {}'.format(rmse_train, mae_train, model.score(X_train, y_train),skill_train))

        try:
            train.sort_values(by='time', inplace=True)
            train.set_index('time', inplace=True)

            range = ('2011-03-01','2011-03-31')
            X_train_sample = train.loc[range[0]:range[1], options.feature_params].astype(np.float32).values
            y_train_sample = train.loc[range[0]:range[1], options.label_params].astype(np.float32).values.ravel()
            times = train.loc[range[0]:range[1], :].index.values.ravel()

            y_pred_sample = model.predict(X_train_sample)
            if options.normalize:
                y_pred_sample = yscaler.inverse_transform(y_pred_sample)

            df = pd.DataFrame(y_pred_sample, index=times)

            # Draw visualisation
            fname='{}/timeseries_training_data_{}_{}.png'.format(options.output_path, range[0], range[1])
            viz.plot_delay(times, y_train_sample, y_pred_sample, 'Train dataset delay', fname)

            fname='scatter_training_data_{}_{}.png'.format(range[0], range[1])
            viz.scatter_predictions(times, y_train_sample, y_pred_sample, savepath=options.output_path, filename=fname)
        except:
            pass

        if options.model == 'llasso':
            print('X_test shape: {}'.format(X_test.shape))
            y_pred, weights = model.predict(X_test, test.loc[:, 'trainstation'].values)
        else:
            y_pred = model.predict(X_test)

        if options.normalize:
            y_pred = yscaler.inverse_transform(y_pred)

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

        logging.info('RMSE: {}'.format(rmse))
        logging.info('MAE: {}'.format(mae))
        logging.info('R2 score: {}'.format(r2))
        logging.info('Brier Skill Score score: {}'.format(skill))

        start = end
        end = start + timedelta(days=int(options.day_step), hours=int(options.hour_step))
        if end > endtime: end = endtime




    # SAVE #####################################################################
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





    # FEATURE SELECTION ########################################################
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
    parser.add_argument('--starttime', type=str, default=None, help='Data starttime')
    parser.add_argument('--endtime', type=str, default=None, help='Data endtime')
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
