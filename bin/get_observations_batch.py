#!/usr/bin/python
import sys
import os
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

from mlfdb import mlfdb as db
# from ml_feature_db.api.mlfdb import mlfdb as db
import lib.io


def split_timerange(starttime, endtime, days=1, hours=0, timestep=60):
    """
    Split timerange to n days

    starttime : Datetime
                starttime
    endtime : Datetime
              endtime
    days : int
           time chunk length, days
    hours : int
           time chunk length, hours
    timestep : int
               timestep in minutes

    return list of tuples [(starttime:Datetime, endtime:Datetime)]
    """
    chunks = []
    start = starttime
    end = start + timedelta(days=days, hours=hours)
    while end <= endtime:
        chunks.append((start + timedelta(minutes=timestep), end))
        start = end
        end = start + timedelta(days=days, hours=hours)
    return chunks

def log_process(count, total):
    """
    Log process
    """
    if count%10 == 0:
        logging.info('Handled {}/{} rows...'.format(count, total))

def get_flashes(args):
    """
    Get flashes for given place

    args : dict
           params to be given in creating url

    return metadata, data (list)
    """

    # Getting flashes is slow. Don't do it for winter times
    d = datetime.fromtimestamp(args['endtime'])
    if int(d.strftime('%m')) < 6 or int(d.strftime('%m')) > 8:
        logging.debug('Not thunder storm season, returning 0 for flashes...')
        return pd.DataFrame()

    url = "http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?param=peak_current&producer=flash&format=ascii&separator=;&starttime={starttime}&endtime={endtime}&latlon={latlon}:{maxdistance}".format(**args)

    #url = "http://data.fmi.fi/fmi-apikey/9fdf9977-5d8f-4a1f-9800-d80a007579c9/timeseries?param=time,peak_current&producer=flash&format=ascii&separator=;&starttime=2017-06-08T00:00:00&endtime=2017-06-08T18:00:00&latlon=57.1324,24.3574:30&timeformat=epoch&tz=local"

    logging.debug('Using url: {}'.format(url))

    rawdata = []
    try:
        with urllib.request.urlopen(url, timeout=300) as u:
            rawdata = u.read().decode("utf-8")
    except Exception as e:
        logging.error(e)

    logging.debug('Flash data loaded')

    if len(rawdata) == 0:
        return pd.DataFrame()

    obsf = StringIO(rawdata)
    obs_df = pd.read_csv(obsf, sep=";", header=None)
    obs_df.rename(columns={0:'time'}, inplace=True)

    return obs_df

def get_ground_obs(params, args):
    """
    Get ground observations which can be fetched directly from
    timeseries

    params : list
             list of params
    args : dict
           params to be given in creating url

    return metadata, data (list)
    """

    url = 'http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?format=ascii&separator=;&producer={producer}&tz=local&timeformat=epoch&latlon={latlon}&starttime={starttime}&endtime={endtime}&timestep=60&param=stationname,{params}&maxdistance={maxdistance}&numberofstations=5'.format(**args)

    logging.debug('Loading data from SmartMet Server...')
    logging.debug('Using url: {}'.format(url))

    with urllib.request.urlopen(url, timeout=300) as u:
        rawdata = u.read().decode("utf-8")
    logging.debug('Observations loaded')

    if len(rawdata) == 0:
        raise ValueError('No data for location {} and time {}'.format(args['latlon'], args['endtime']))

    obsf = StringIO(rawdata)
    obs_df = pd.read_csv(obsf, sep=";", header=None)
    obs_df.rename(columns={1:'time'}, inplace=True)

    return obs_df

def process_timerange(starttime, endtime, params, param_names, producer):

    """
    Process timerange
    """
    #a = mlfdb.mlfdb(options.db_config_file)
    a = db.mlfdb(options.db_config_file)
    io = lib.io.IO()
    process_id = os.getpid()

    startstr = starttime.strftime('%Y-%m-%dT%H:%M:%S')
    endstr = endtime.strftime('%Y-%m-%dT%H:%M:%S')

    logging.info('Process {}: Processing time {} - {}...'.format(process_id, startstr, endstr))

    # Get labels, features and filter labels that do not have features
    try:
        l_metadata, l_header, l_data = a.get_rows(options.dataset, rowtype='label', starttime=starttime, endtime=endtime)
        f_metadata, f_header, f_data = a.get_rows(options.dataset, rowtype='feature', starttime=starttime, endtime=endtime)
        filt_metadata, _ = io.filter_labels(l_metadata, l_data, f_metadata, f_data, invert=True, uniq=True)
    except ValueError as e:
        logging.debug(e)
        return 0

    latlons = set()
    for row in filt_metadata:
        latlons.add(str(row[3])+','+str(row[2]))

    logging.info('Process {}: There are {} (of total {}) lines to process...'.format(process_id, len(filt_metadata), len(l_metadata)))

    if len(filt_metadata) == 0:
        return 0

    # Create url and get data
    apikey = '9fdf9977-5d8f-4a1f-9800-d80a007579c9'
    count = 0
    placecount = 0
    for latlon in latlons:
        placecount += 1
        label_metadata = io.filter_labels_by_latlon(filt_metadata, latlon)
        starttime, endtime = io.get_timerange(label_metadata)
        startstr = starttime.strftime('%Y-%m-%dT%H:%M:%S')
        endstr = endtime.strftime('%Y-%m-%dT%H:%M:%S')
        prec_start = starttime - timedelta(hours=6)

        args = {
            'latlon': latlon,
            'params': ','.join(params),
            'starttime' : prec_start.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': endstr,
            'producer': options.producer,
            'apikey' : apikey,
            'maxdistance' : 100000
        }

        flash_start = starttime - timedelta(hours=1)
        flash_args = {
            'starttime' : flash_start.timestamp(), #strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime' : endtime.timestamp(),
            'latlon' : latlon,
            'apikey' : apikey,
            'maxdistance' : 30 # Harmonie FlashMultiplicity value in km
        }


        def get_obs(params, args, flash_args, label_metadata, i=0):
            i += 1
            try:
                obs_df = get_ground_obs(params, args)
                obs_df = io.find_best_station(obs_df)
                metadata_df, data_df = io.filter_ground_obs(obs_df, label_metadata)

                flash_df = get_flashes(flash_args)
                data_df = io.filter_flashes(flash_df, data_df)

                data_df = io.filter_precipitation(data_df, prec_column=23)
                data_df.fillna(-99, inplace=True)
            except (urllib.error.URLError, timeout, ConnectionResetError) as e:
                logging.error('Process {}: Timeout. Trying again ({}/5)...'.format(process_id, i))
                logging.error('Process {}: Exception: {}'.format(process_id, e))
                # logging.error('Params: {}'.format(args))
                # logging.error('Flash params: {}'.format(flahs_args))
                if i < 5:
                    metadata_df, data_df = get_obs(params, args, flash_args, label_metadata, i)
                else:
                    raise urllib.error.URLError()

            return metadata_df, data_df

        try:
            metadata_df, data_df = get_obs(params, args, flash_args, label_metadata)
        except ValueError as e:
            logging.error(e)
            continue

        data = np.array(data_df.drop(columns=['time', 2])).astype(np.float)
        metadata = metadata_df.as_matrix()

        logging.debug('Dataset size: {}'.format(data.shape))
        logging.debug('Metadata size: {}'.format(np.array(metadata).shape))

        def add_rows(header, data, metadata, dataset, i=0):
            i += 1
            try:
                count = a.add_rows('feature', header, data, metadata, dataset, time_column=1, loc_column=0)
            except psycopg2.OperationalError:
                logging.error('Connection error to db. Trying again ({}/5)...'.format(i))
                if i < 5:
                    count = add_rows(header, data, metadata, dataset, i)
                else:
                    raise psycopg2.OperationalError()

            return count

        if len(data) > 0:
            # Save to database
            header = param_names[2:] + ['count_flash', 'precipitation3h', 'precipitation6h']
            count += add_rows(header, data, metadata, options.dataset)

        if placecount%10 == 0:
            logging.info('Process {}: {}/{} locations processed ({} rows inserted)...'.format(process_id, placecount, len(latlons), count))

    return count

def main():
    """
    Get observations near locations from SmartMet Server

    Data start and end time and timestep is fetched from the
    data. Dataset is assumed coherent in means of time and
    locations. I.e. timestep is assumed to be constant between start
    and end time.
    """

    # Initialize process pool
    job_count = options.job_count
    if job_count < 0:
        job_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(job_count)

    a = db.mlfdb(options.db_config_file)
    io = lib.io.IO()
    params, param_names = io.read_parameters('cnf/parameters.txt')

    if options.replace:
        logging.info('Removing old dataset...')
        a.remove_dataset(options.dataset, type='feature')

    # Split time range to chunks and process them in paraller
    start = datetime.strptime(options.starttime, '%Y-%m-%d')
    end = datetime.strptime(options.endtime, '%Y-%m-%d')

    res = [pool.apply_async(process_timerange, args=(chunk[0], chunk[1], params, param_names, options.producer)) for chunk in split_timerange(start, end, days=0, hours=12)]
    total_count = [p.get() for p in res]

    logging.info('Added {} rows observations to db.'.format(sum(total_count)))

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_count',
                        type=int,
                        default=1,
                        help='Job count for processing. Use -1 for using all cores')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='Name of dataset bind to locations')
    parser.add_argument('--distance',
                        type=str,
                        default=50,
                        help='How far from station locations are searched for')
    parser.add_argument('--starttime',
                        type=str,
                        default=None,
                        help='Start time')
    parser.add_argument('--endtime',
                        type=str,
                        default=None,
                        help='End time')
    parser.add_argument('--timestep',
                        type=str,
                        default=10,
                        help='Timestep of observations in minutes')
    parser.add_argument('--producer',
                        type=str,
                        default='fmi',
                        help='Data producer')
    parser.add_argument('--replace',
                        action='store_true',
                        help='If set, old dataset is removed first, default=False')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')
    parser.add_argument('--db_config_file',
                        type=str,
                        default=None,
                        help='GS address for db config file (if none, ~/.mlfdbconfig is used)')


    options = parser.parse_args()

    debug=False

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
