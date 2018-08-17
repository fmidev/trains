import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import numpy as np
import pandas as pd

from lib import bqhandler as _bq
from lib import imputer

def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()

    times = []
    # times.append({'starttime': dt.datetime.strptime('2009-11-29', "%Y-%m-%d"),
    #               'endtime': dt.datetime.strptime('2018-01-10', "%Y-%m-%d")})
    times.append({'starttime': dt.datetime.strptime('2014-06-02', "%Y-%m-%d"),
                  'endtime': dt.datetime.strptime('2018-01-10', "%Y-%m-%d")})

    logging.info('Using times: {}'.format(times))

    #scaler = StandardScaler()
    data_to_scale = pd.DataFrame()

    daystep = 90
    for t in times:
        starttime = t['starttime']
        endtime = t['endtime']

        start = starttime
        end = start + timedelta(days=daystep)
        if end > endtime: end = endtime

        while end <= endtime and start < end:

            logging.info('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'),
                                                                end.strftime('%Y-%m-%d %H:%M')))

            logging.info('Reading data...')
            try:
                data = bq.get_rows(start,
                                   end,
                                   project=options.project,
                                   dataset=options.src_dataset,
                                   table=options.src_table)

                logging.info('Imputing missing values...')
                data = imputer.fit_transform(data)
                data_to_scale = pd.concat([data_to_scale, data])

                data.set_index(['time', 'trainstation'], inplace=True)

                if len(data) < 1 or len(data) < 1:
                    start = end
                    end = start + timedelta(days=daystep)
                    continue

                bq.dataset_to_table(data, options.dst_dataset, options.dst_table)

            except ValueError as e:
                logging.warning(e)

            start = end
            end = start + timedelta(days=daystep)
            if end > endtime: end = endtime


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='trains-197305', help='BigQuery project name')
    parser.add_argument('--src_dataset', type=str, default='trains_2009_18_wo_testset', help='Src dataset name for features')
    parser.add_argument('--src_table', type=str, default='features_1', help='Src table name for features')
    parser.add_argument('--dst_dataset', type=str, default='trains_2009_18_wo_testset', help='Dst dataset name for features')
    parser.add_argument('--dst_table', type=str, default='features_imputed', help='Dst table name for features')

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
