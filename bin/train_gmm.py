import sys,os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import argparse, logging, json
import datetime as dt
from datetime import timedelta

import itertools
from collections import OrderedDict

import math
import numpy as np
import pandas as pd

from sklearn import mixture
from sklearn import metrics

import lib.io as _io
import lib.viz as _viz
import lib.bqhandler as _bq
import lib.config as _config


def main():
    """
    Main program
    """
    bq = _bq.BQHandler()
    io = _io.IO(gs_bucket=options.gs_bucket)
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    #save_path = options.save_path+'/'+options.config_name

    logging.info('Using dataset {}.{} and time range {} - {}'.format(options.feature_dataset,
                                                                     options.feature_table,
                                                                     starttime.strftime('%Y-%m-%d'),
                                                                     endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    logging.info('Reading data...')
    bq.set_params(batch_size=2500000,
                  loc_col='trainstation',
                  project=options.project,
                  dataset=options.feature_dataset,
                  table=options.feature_table,
                  parameters=all_param_names,
                  only_winters =options.only_winters,
                  reason_code_table=options.reason_code_table)

    data = bq.get_rows(starttime, endtime)

    data = io.filter_train_type(labels_df=data,
                                train_types=options.train_types,
                                sum_types=True,
                                train_type_column='train_type',
                                location_column='trainstation',
                                time_column='time',
                                sum_columns=['train_count','delay'],
                                aggs=aggs)

    if options.y_avg:
        data = io.calc_delay_avg(data)

    data.sort_values(by=['time', 'trainstation'], inplace=True)

    if options.normalize:
        logging.info('Normalizing data...')
        xscaler = StandardScaler()
        yscaler = StandardScaler()

        non_scaled_data = data.loc[:,options.meta_params]
        labels = data.loc[:, options.label_params].values.reshape((-1, 1))

        yscaler.fit(labels)
        scaled_labels = pd.DataFrame(yscaler.transform(labels), columns=['delay'])
        scaled_features = pd.DataFrame(xscaler.fit_transform(data.loc[:,options.feature_params]),
                                           columns=options.feature_params)

        data = pd.concat([non_scaled_data, scaled_features, scaled_labels], axis=1)

    if options.pca:
        logging.info('Doing PCA analyzis for the data...')
        ipca = IncrementalPCA(n_components=options.pca_components,
                              whiten = options.whiten,
                              copy = False)

        non_processed_data = data.loc[:,options.meta_params+ options.label_params]
        processed_data = data.loc[:, options.feature_params]
        ipca.fit(processed_data)
        processed_features = pd.DataFrame(ipca.transform(processed_data))

        data = pd.concat([non_processed_data, processed_data], axis=1)

        fname = options.output_path+'/ipca_explained_variance.png'
        viz.explained_variance(ipca, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)


    d_full = data.loc[:, options.feature_params].dropna()
    logging.info('Data dimensions: {}'.format(data.shape))

    logging.info('Training...')
    gmm = mixture.GaussianMixture(n_components=4, covariance_type='full')
    gmm.fit(d_full)

    if options.normalize:
        fname = options.save_path+'/yscaler.pkl'
        io.save_scikit_model(yscaler, fname, fname)

    fname = options.save_path+'/gmm.pkl'
    io.save_scikit_model(gmm, fname)

    bic = gmm.bic(d_full)
    aic = gmm.aic(d_full)
    llavg = gmm.score(d_full)
    perfstr = 'BIC: {:.4f} | AIC: {:.4f} | AVG LL: {:.4f}'.format(bic, aic, llavg)
    logging.info(perfstr)

    fname = options.output_path+'/gmm_prediction_hist_trainset.png'
    viz.hist(gmm.predict(d_full), 'Sample count', title=perfstr, filename=fname)

    # Ensure that evertyhing is saved
    io._upload_dir_to_bucket(options.save_path, options.save_path)
    io._upload_dir_to_bucket(options.output_path, options.output_path)

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
