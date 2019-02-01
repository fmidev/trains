import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np

from lib import io as _io
from lib import bqhandler as _bq

def main():
    """
    Get data from db and save it as csv
    """

    bq = _bq.BQHandler()
    io = _io.IO()

    times = []
    times.append({'starttime': dt.datetime.strptime('2011-02-01', "%Y-%m-%d"),
                  'endtime': dt.datetime.strptime('2011-03-01', "%Y-%m-%d")
                 })

    times.append({'starttime': dt.datetime.strptime('2016-06-01', "%Y-%m-%d"),
                  'endtime': dt.datetime.strptime('2016-07-01', "%Y-%m-%d")
    })

    times.append({'starttime': dt.datetime.strptime('2017-02-01', "%Y-%m-%d"),
                  'endtime': dt.datetime.strptime('2017-03-01', "%Y-%m-%d")})

    logging.info('Using times: {}'.format(times))

    for t in times:
        start = t['starttime']
        end = t['endtime']

        logging.info('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'),
                                                            end.strftime('%Y-%m-%d %H:%M')))

        logging.info('Reading data...')
        data = bq.get_rows(start,
                           end,
                           project=options.project,
                           dataset=options.src_dataset,
                           table=options.src_table)

        #print(data.shape
        data.set_index(['time', 'trainstation'], inplace=True)
        #print(data)
        bq.dataset_to_table(data, options.dst_dataset, options.dst_table)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='trains-197305', help='BigQuery project name')
    parser.add_argument('--src_dataset', type=str, default='trains_data', help='Src dataset name for features')
    parser.add_argument('--src_table', type=str, default='features_imputed', help='Src table name for features')
    parser.add_argument('--dst_dataset', type=str, default='trains_data', help='Dst dataset name for features')
    parser.add_argument('--dst_table', type=str, default='features_imputed_testset', help='Dst table name for features')

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
