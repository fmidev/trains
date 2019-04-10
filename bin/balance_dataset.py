import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np
import pandas as pd

from sklearn.utils import class_weight

from lib import io as _io
from lib import bqhandler as _bq

def log_class_dist(data):
    """
    Log class distributions
    """
    c0 = sum((data < 1))
    c1 = sum((data > 0) & (data < 2))
    c2 = sum((data > 1) & (data < 3))
    c3 = sum((data > 2))
    c_all = len(data)
    logging.info('Class sizes: 0: {} ({:.02f}%), 1: {} ({:.02f}%), 2: {} ({:.02f}%), 3: {} ({:.02f}%)'.format(c0, c0/c_all*100, c1, c1/c_all*100, c2, c2/c_all*100, c3, c3/c_all*100))

def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()
    io = _io.IO()

    starttime = dt.datetime.strptime('2010-01-01', "%Y-%m-%d")
    endtime = dt.datetime.strptime('2019-01-01', "%Y-%m-%d")

    logging.info('Reading data...')
    data = bq.get_rows(starttime,
                       endtime,
                       project=options.project,
                       dataset=options.src_dataset,
                       table=options.src_table)

    logging.info('Data loaded.')
    #print(data.shape
    data = io.calc_delay_avg(data)
    data = io.classify(data)
    log_class_dist(data.loc[:,'class'])


    count = data.groupby('class').size().min()
    balanced_data = pd.concat([data.loc[data['class'] == 0].sample(n=count),
                               data.loc[data['class'] == 1].sample(n=count),
                               data.loc[data['class'] == 2].sample(n=count),
                               data.loc[data['class'] == 3].sample(n=count)])
    print(balanced_data.head(5))
    print(balanced_data.groupby('class').size())

    balanced_data.set_index(['time', 'trainstation'], inplace=True)

    logging.info('Saving data...')
    #print(data)
    bq.dataset_to_table(balanced_data, options.dst_dataset, options.dst_table)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='trains-197305', help='BigQuery project name')
    parser.add_argument('--src_dataset', type=str, default='trains_data', help='Src dataset name for features')
    parser.add_argument('--src_table', type=str, default='features_wo_testset', help='Src table name for features')
    parser.add_argument('--dst_dataset', type=str, default='trains_data', help='Dst dataset name for features')
    parser.add_argument('--dst_table', type=str, default='balanced_features_wo_testset', help='Dst table name for features')

    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    options = parser.parse_args()

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
