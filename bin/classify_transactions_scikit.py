import sys, os
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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
    viz = Viz(io)

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    if options.model == 'bgm':
        model = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process",
                                        n_components=options.n_components)
    elif options.model == 'gaussiannb':
        model = GaussianNB()
    elif options.model == 'rfc':
        model = RandomForestClassifier(n_jobs=-1)
    elif options.model == 'svc':
        params = {'shrinking': True, 'kernel': 'rbf', 'gamma': 0.5, 'C': 1, 'probability': True}
        model = SVC(**params)
    else:
        raise('Model not specificied or wrong. Add for example "model: bgm" to config file.')

    if options.pca:
        ipca = IncrementalPCA(n_components=options.pca_components,
                              whiten = options.whiten,
                              copy = False)

    sum_columns = ['delay']
    if options.reason_code_table is not None:
        sum_columns = ['count']

    logging.info('Reading data...')
    data = bq.get_rows(starttime,
                       endtime,
                       loc_col='trainstation',
                       project=options.project,
                       dataset=options.feature_dataset,
                       table=options.feature_table,
                       parameters=all_param_names,
                       reason_code_table=options.reason_code_table,
                       only_winters=options.only_winters)

    data = io.filter_train_type(labels_df=data,
                                train_types=options.train_types,
                                sum_types=True,
                                train_type_column='train_type',
                                location_column='trainstation',
                                time_column='time',
                                sum_columns=sum_columns,
                                aggs=aggs)

    # Sorting is actually not necessary. It's been useful for debugging.
    data.sort_values(by=['time', 'trainstation'], inplace=True)
    data.set_index('time', inplace=True)

    logging.info('Data contain {} rows...'.format(len(data)))

    logging.info('Adding binary class to the dataset with limit {}...'.format(options.delay_limit))
    #logging.info('Adding binary class to the dataset with limit {}...'.format(limit))
    #data['class'] = data['count'].map(lambda x: 1 if x > options.delay_count_limit else -1)
    data['class'] = data['delay'].map(lambda x: 1 if x > options.delay_limit else -1)
    io.log_class_dist(data.loc[:, 'class'].values, labels=[-1,1])

    if options.balance:
        logging.info('Balancing dataset...')
        count = data.groupby('class').size().min()
        data = pd.concat([data.loc[data['class'] == -1].sample(n=count),
                          data.loc[data['class'] == 1].sample(n=count)])
        io.log_class_dist(data.loc[:, 'class'].values, labels=[-1,1])

    if options.month:
        logging.info('Adding month to the dataset...')
        data['month'] = data.index.map(lambda x: x.month)
        options.feature_params.append('month')

    target = data.loc[:,'class'].astype(np.int32).values.ravel()
    features = data.loc[:,options.feature_params].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.10)

    if options.normalize:
        logging.info('Normalizing data...')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
        io.report_cv_results(random_search.cv_results_, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)
        sys.exit()
    else:
        logging.info('Training...')
        model.fit(X_train, y_train)

        # Save model and xscaler (no reason to save xscaler before the model has fitted as well)
        io.save_scikit_model(model, filename=options.save_file, ext_filename=options.save_file)
        if options.normalize:
            fname=options.save_path+'/xscaler.pkl'
            io.save_scikit_model(scaler, filename=fname, ext_filename=fname)

    # Metrics
    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
     # We want [-1,1] classes as y values are
    y_pred[y_pred == 0] = -1

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    logging.info('Accuracy: {}'.format(acc))
    logging.info('Precision: {}'.format(precision))
    logging.info('Recall: {}'.format(recall))
    logging.info('F1 score: {}'.format(f1))
    io.log_class_dist(y_pred, labels=[-1,1])

    error_data = {'acc': [acc],
                  'precision': [precision],
                  'recall': [recall],
                  'f1': [f1]}
    fname = '{}/training_time_validation_errors.csv'.format(options.output_path)
    io.write_csv(error_data, filename=fname, ext_filename=fname)

    # Confusion matrices
    fname = '{}/confusion_matrix_validation.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(2), filename=fname)

    fname = '{}/confusion_matrix_validation_normalised.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(2), True, filename=fname)

    # Precision-recall curve
    fname = '{}/precision-recall-curve.png'.format(options.output_path)
    viz.prec_rec_curve(y_test, y_pred_proba, filename=fname)

    # ROC
    fname = '{}/roc.png'.format(options.output_path)
    viz.plot_binary_roc(y_test, y_pred_proba, filename=fname)


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

    logging.info('Using configuration: {} | {}'.format(options.config_filename, options.config_name))

    main()
