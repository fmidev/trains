#!/usr/bin/python
import sys
import argparse
import logging
from datetime import datetime, timedelta
import time
import json
import itertools
import numpy as np
import urllib.request, json
from socket import timeout
import requests
import codecs
import multiprocessing
from io import StringIO
import pandas as pd

from mlfdb import mlfdb
import lib.io
from lib import dbhandler as _db

def get_forecasts(args):
    """
    Get forecasts

    args : dict
           params to be given in creating url

    return pandas df
    """

    url = 'http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?format=ascii&separator=;&producer={producer}&tz=local&timeformat=epoch&endtime=data&latlons={latlons}&param={params}'.format(**args)

    logging.debug('Loading data from SmartMet Server...')
    logging.info('Using url: {}'.format(url))

    with urllib.request.urlopen(url, timeout=6000) as u:
        rawdata = args['params'].replace(',',';')+'\n'+u.read().decode("utf-8")
    logging.debug('Data loaded')

    obsf = StringIO(rawdata)
    df = pd.read_csv(obsf, sep=";")

    return df

def main():
    """
    Get forecasted delay for every station
    """

    io = lib.io.IO()

    params, _ = io.read_parameters(options.parameters_filename)
    params.append('origintime')
    stations = io.get_train_stations(filename=options.stations_filename)

    max_count = 5000
    if options.dev == 1: max_count=1

    latlons = []
    names = {}
    for name, station in stations.items():
        if len(latlons) >= max_count:
            break
        latlon = str(station['lat'])+','+str(station['lon'])
        latlons.append(latlon)
        names[latlon] = name

    logging.info('Getting delay forecast for {} locations...'.format(len(stations)))

    # Create url and get data
    apikey = 'a13287ed-007d-476f-bfe3-80bf5e8ea8a0'

    args = {
        'latlons': ','.join(latlons),
        'params': ','.join(params),
        'producer': 'harmonie_skandinavia_pinta',
        'apikey' : apikey
    }

    logging.info('Getting forecast...')
    data = get_forecasts(args)
    logging.info('Calculating precipitation sums...')
    data = io._calc_prec_sums(data, prec_column='PrecipitationInstantTotal').fillna(-99)
    logging.info('Data shape: {}'.format(data.shape))

    files = io.df_to_serving_file(data)

    result = io.predict_gcloud_ml('trains_lr',
                                  'tiny_subset_8',
                                  files,
                                  data,
                                  names)
    logging.info('Got predictions for {} stations. First station has {} values.'.format(len(result), len(next(iter(result.values())))))

    if options.dev == 0:
        # Insert into db
        logging.info('Inserting results into db...')
        db = _db.DBHandler()
        db.insert_forecasts(result)

        # Clean db
        logging.info('Cleaning db...')
        tolerance = timedelta(days=2)
        db.clean_forecasts(tolerance)
    else:
        logging.info('Results are: {}'.format(result))

    logging.info('Done')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--stations_filename',
                        type=str,
                        default='cnf/stations.json',
                        help='Stations file name')
    parser.add_argument('--parameters_filename',
                        type=str,
                        default='cnf/forecast_parameters.txt',
                        help='Parameters file name')
    parser.add_argument('--dev',
                        type=int,
                        default=0,
                        help='Set 1 for development mode')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    options = parser.parse_args()

    debug=False

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
