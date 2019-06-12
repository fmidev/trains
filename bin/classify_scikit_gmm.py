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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.mixture import BayesianGaussianMixture

from sklearn.metrics import accuracy_score, f1_score #, precisision_score, recall_score

from lib import io as _io
from lib import viz as _viz
from lib import bqhandler as _bq
from lib import imputer
from lib import config as _config

def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {}.{} and time range {} - {}'.format(options.feature_dataset,
                                                                     options.feature_table,
                                                                     starttime.strftime('%Y-%m-%d'),
                                                                     endtime.strftime('%Y-%m-%d')))

    all_param_names = list(set(options.label_params + options.feature_params + options.meta_params + options.gmm_params))
    aggs = io.get_aggs_from_param_names(options.feature_params)

    if options.model == 'bgm':
        model = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process",
                                        n_components=options.n_components)

    elif options.model == 'rfc':
        raise('Not implemented. Get back to work!')
    else:
        raise('Model not specificied or wrong. Add "model: bgm" to config file.')

    if options.pca:
        ipca = IncrementalPCA(n_components=options.pca_components,
                              whiten = options.whiten,
                              copy = False)

    logging.info('Processing time range {} - {}'.format(starttime.strftime('%Y-%m-%d %H:%M'),
                                                        endtime.strftime('%Y-%m-%d %H:%M')))

    logging.info('Reading data...')
    data = bq.get_rows(starttime,
                       endtime,
                       loc_col='trainstation',
                       project=options.project,
                       dataset=options.feature_dataset,
                       table=options.feature_table,
                       parameters=all_param_names)
    # data = io.filter_train_type(labels_df=data,
    #                             train_types=options.train_types,
    #                             sum_types=True,
    #                             train_type_column='train_type',
    #                             location_column='trainstation',
    #                             time_column='time',
    #                             sum_columns=options.label_params,
    #                             aggs=aggs)

    data.sort_values(by=['time', 'trainstation'], inplace=True)

    logging.info('Processing {} rows...'.format(len(data)))

    io.log_class_dist(data.loc[:, 'class'].values, 4)

    if options.month:
        logging.info('Adding month to the dataset...')
        data['month'] = data['time'].map(lambda x: x.month)
        options.feature_params.append('month')

    # GMM Classification
    logging.info('Doing GMM Classification...')
    gmm_model = io.load_scikit_model(options.gmm_classifier)
    gmm_data = data.loc[:, options.gmm_params]
    data['gmm_class'] = gmm_model.predict(gmm_data.values)
    options.feature_params.append('gmm_class')

    # Extract targets and features
    target = data.loc[:,options.label_params].astype(np.int32).values
    features = data.loc[:,options.feature_params].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.10)

    logging.debug('Features shape: {}'.format(X_train.shape))

    n_samples, n_dims = X_train.shape

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
        if options.model == 'bgm':
            param_grid = {"n_components": [1, 2, 4, 8, 16],
                          "covariance_type": ['full', 'tied', 'diag', 'spherical'],
                          "init_params": ['kmeans', 'random']
                          }
        elif options.model == 'rfc':
            raise("Not implemented. Get back to work!")
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
        # TODO this is where I left
        logging.info('Training...')
        model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info('Accuracy: {}'.format(acc))

    io.log_class_dist(y_pred, 4)

    io.save_scikit_model(model, filename=options.save_file, ext_filename=options.save_file)

    # if options.model == 'rf':
    #     fname = options.output_path+'/rfc_feature_importance.png'
    #     viz.rfc_feature_importance(model.feature_importances_, fname, feature_names=options.feature_params)
    #     io._upload_to_bucket(filename=fname, ext_filename=fname)
    #
    # try:
    #     fname = options.output_path+'/learning_over_time.png'
    #     viz.plot_learning_over_time(end_times_obj, rmses, maes, r2s, filename=fname)
    #     io._upload_to_bucket(filename=fname, ext_filename=fname)
    # except Exception as e:
    #     logging.error(e)
    #
    # error_data = {'start_times': start_times,
    #               'end_times': end_times,
    #               'rmse': rmses,
    #               'mae': maes,
    #               'r2': r2s}
    # fname = '{}/training_time_validation_errors.csv'.format(options.output_path)
    # io.write_csv(error_data, filename=fname, ext_filename=fname)

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
