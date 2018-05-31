import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np
import pandas as pd
# import tensorflow as tf
from mlfdb import mlfdb
from ml_feature_db.api.mlfdb import mlfdb as db
from lib import io as _io
from lib import viz as _viz

def main():
    """
    """

    #a = mlfdb.mlfdb()
    a = db.mlfdb()
    io = _io.IO()
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    logging.info('Using time range {} - {}'.format(starttime.strftime('%Y-%m-%d'), endtime.strftime('%Y-%m-%d')))

    params, param_names = io.read_parameters('cnf/parameters.txt', drop=2)

    day_step = 30
    hour_step = 0

    start = starttime
    end = start + timedelta(days=day_step, hours=hour_step)
    if end > endtime: end = endtime

    while end <= endtime:
        logging.info('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M')))

        try:
            data = a.get_rows(options.src_dataset,
                              starttime=start,
                              endtime=end,
                              rowtype='feature',
                              return_type='pandas'),
                              parameters=param_names)

        except ValueError as e:
            print(e)
            start = end
            end = start + timedelta(days=day_step, hours=hour_step)
            continue

        print(data)
        #logging.debug('Features metadata shape: {} | Features shape: {}'.format(f_metadata.shape, f_data.shape))

        logging.info('Processing {} rows...'.format(len(f_data)))

        start = end
        end = start + timedelta(days=day_step, hours=hour_step)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, help='End time of the classification data interval')
    parser.add_argument('--output_path', type=str, default=None, help='Output save path')
    parser.add_argument('--src_dataset', type=str, default=None, help='Source dataset name')
    parser.add_argument('--dst_dataset', type=str, default=None, help='Destination dataset name')
    parser.add_argument('--db_config_file',
                        type=str,
                        default=None,
                        help='GS address for db config file (if none, ~/.mlfdbconfig is used)')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    options = parser.parse_args()

    if options.output_path is not None and not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
