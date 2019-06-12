import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
# import json
# import itertools
import numpy as np
from configparser import ConfigParser
import gpflow
import tensorflow as tf

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq
from lib import imputer
from lib import config as _config
from lib.gp import GP

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

    print(res)

def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    print('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)


    if options.pca:
        ipca = IncrementalPCA(n_components=options.pca_components,
                              whiten = options.whiten,
                              copy = False)

    rmses, maes, r2s, vars, start_times, end_times, end_times_obj = [], [], [], [], [], [], []

    start = starttime
    end = endtime
    print('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'),
                                                 end.strftime('%Y-%m-%d %H:%M')))

    try:
        print('Reading data...')
        data = bq.get_rows(start,
                           end,
                           loc_col='trainstation',
                           project=options.project,
                           dataset=options.feature_dataset,
                           table=options.feature_table,
                           parameters=all_param_names)
        data = io.filter_train_type(labels_df=data,
                                    train_types=options.train_types,
                                    sum_types=True,
                                    train_type_column='train_type',
                                    location_column='trainstation',
                                    time_column='time',
                                    sum_columns=['delay'],
                                    aggs=aggs)

        if options.y_avg_hours is not None:
            data = io.calc_running_delay_avg(data, options.y_avg_hours)

        data.sort_values(by=['time', 'trainstation'], inplace=True)

        if options.impute:
            print('Imputing missing values...')
            data.drop(columns=['train_type'], inplace=True)
            data = imputer.fit_transform(data)
            data.loc[:, 'train_type'] = None

        if options.model == 'ard' and len(data) > options.n_samples:
            print('Sampling {} values from data...'.format(options.n_samples))
            data = data.sample(options.n_samples)

        #l_data = data.loc[:,options.meta_params + options.label_params]
        #f_data = data.loc[:,options.meta_params + options.feature_params]

    except ValueError as e:
        f_data, l_data = [], []

    #f_data.rename(columns={'trainstation':'loc_name'}, inplace=True)

    #logging.debug('Labels shape: {}'.format(l_data.shape))
    print('Processing {} rows...'.format(len(data)))
    #assert l_data.shape[0] == f_data.shape[0]

    target = data.loc[:, options.label_params].astype(np.float32).values
    #print(f_data.columns)
    #features = f_data.drop(columns=['loc_name', 'time']).astype(np.float32).values
    features = data.loc[:, options.feature_params].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33)

    logging.debug('Features shape: {}'.format(X_train.shape))

    n_samples, n_dims = X_train.shape

    if options.normalize:
        print('Normalizing data...')
        print(X_train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

    if options.pca:
        print('Doing PCA analyzis for the data...')
        X_train = ipca.fit_transform(X_train)
        fname = options.output_path+'/ipca_explained_variance.png'
        viz.explained_variance(ipca, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)
        X_test = ipca.fit_transform(X_test)

    logging.debug('Features shape after pre-processing: {}'.format(X_train.shape))

    print('Training...')
    print(X_train.shape)
    input_dim = X_train.shape[1]
    #k1 = gpflow.kernels.Matern52(input_dim, lengthscales=0.3)
    #k_seasonal = gpflow.kernels.Periodic(input_dim=input_dim, period=2190, name='k_seasonal')
    #k_small = gpflow.kernels.Periodic(input_dim=input_dim, period=120, name='k_small')
    k_weather = gpflow.kernels.RBF(input_dim=input_dim, ARD=True)
    #k_noise = gpflow.kernels.White(input_dim=input_dim)

    #k = k_seasonal + k_weather + k_noise
    k = k_weather
    Z = np.random.rand(150, input_dim)

    if options.cv:
        logging.info('Doing random search for hyper parameters...')

        param_grid = {"length_scale": [0.1, 1, 2],
                      "whiten": [True, False]
                      }

        model = GP(dim=input_dim,
                   Z=Z)

        random_search = RandomizedSearchCV(model,
                                           param_distributions=param_grid,
                                           n_iter=int(options.n_iter_search),
                                           n_jobs=-1)

        random_search.fit(X_train, y_train)
        logging.info("RandomizedSearchCV done.")
        sys.exit()
    else:
        model = GP(dim=input_dim,
                   Z=Z
                   )
        model.fit(X_train.astype(np.float64),
                  y_train.reshape((-1,1)).astype(np.float64))

        model.save(options.save_file)

        print('Training finished')
        print(model.model)

#    Z_list = options.z_list.split(',')

    #for size in Z_list:

    #    with tf.Session() as sess:
            #custom_config = gpflow.settings.get_settings()
            #custom_config.verbosity.tf_compile_verb = True

            #with gpflow.settings.temp_settings(custom_config), gpflow.session_manager.get_session().as_default():

            #Z = X_train[::5].copy()
            # Z = np.random.rand(int(size), 19)
            # print('Training with inducing points: {}'.format(Z.shape))
            #
            # # model = gpflow.models.SVGP(X_train.astype(np.float64),
            # #                            y_train.reshape((-1,1)).astype(np.float64),
            # #                            kern=k,
            # #                            likelihood=gpflow.likelihoods.Gaussian(),
            # #                            Z=Z,
            # #                            #Z=X_train.copy(),
            # #                            minibatch_size=100,
            # #                            whiten=options.normalize
            # #                            )
            # #                            #model.likelihood.variance = 0.01
            # #
            # # model.compile(session=sess)
            # # opt = gpflow.train.ScipyOptimizer()
            # # opt.minimize(model)
            #
            # model = GP(dim=19,
            #            Z=Z
            #            )
            # model.fit(X_train.astype(np.float64),
            #           y_train.reshape((-1,1)).astype(np.float64))
            #
            # model.save(options.save_file)
            #
            # print('Training finished')
            # print(model.model)

            #fname=options.output_path+'/svga_performance.png'
            #viz.plot_svga(model, fname)

            # k_long_term = 66.0**2 * RBF(length_scale=67.0)
            # k_seasonal = 2.4**2 * RBF(length_scale=90.0)* ExpSineSquared(length_scale=150, periodicity=1.0, periodicity_bounds=(0,10000))
            # k_medium_term = 0.66**2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
            # k_noise = 0.18**2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19**2)
            # #kernel_gpml = k_long_term + k_seasonal + k_medium_term + k_noise
            # kernel_gpml = k_long_term + k_seasonal + k_medium_term + k_noise
            #
            # model = GaussianProcessRegressor(kernel=kernel_gpml, #alpha=0,
            #                                  optimizer=None, normalize_y=True)

        # Metrics
        y_pred, var = model.predict_f(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        vars.append(var.mean())
        start_times.append(start.strftime('%Y-%m-%dT%H:%M:%S'))
        end_times.append(end.strftime('%Y-%m-%dT%H:%M:%S'))
        end_times_obj.append(end)

        print('RMSE: {:.2f}'.format(rmse))
        print('MAE: {:.2f}'.format(mae))
        print('Variance: {:.2f}-{:.2f}'.format(var.min(), var.max()))
        print('R2 score: {:.2f}'.format(r2))


    #io.save_scikit_model(model, filename=options.save_file, ext_filename=options.save_file)
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
                  'var': vars,
                  'r2': r2s}
    fname = '{}/training_time_validation_errors.csv'.format(options.output_path)
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

    _config.read(options)

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}

    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
