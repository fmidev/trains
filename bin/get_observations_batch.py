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

def get_latlons(a, dataset):
    """
    Get latlons
    
    dataset : str
              dataset name
    
    return : list, struct
             list of latlons, struct of ids ids{'latlon': xx}
    """
    latlons = []
    ids = dict()
    locations = a.get_locations_by_dataset(dataset)
    for loc in locations:
        latlon = str(loc[3])+','+str(loc[2])
        latlons.append(latlon)
        ids[latlon] = loc[0]
        
    return latlons, ids


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

def get_prec_sum(args):
    """ 
    Get precipitation sum
    """
    url = "http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?format=ascii&separator=;&producer={producer}&tz=local&timeformat=epoch&latlon={latlon}&timestep=60&starttime={starttime}&endtime={endtime}&param=stationname,time,place,lat,lon,precipitation1h&maxdistance={maxdistance}&numberofstations=5".format(**args)

    logging.debug('Using url: {}'.format(url))
    
    with urllib.request.urlopen(url, timeout=600) as u:
        rawdata = u.read().decode("utf-8")

    logging.debug('Precipitation data loaded')
    
    if len(rawdata) == 0:
        return pd.DataFrame()

    obsf = StringIO(rawdata)    
    obs_df = pd.read_csv(obsf, sep=";", header=None)
    obs_df.rename(columns={1:'time'}, inplace=True)
    return obs_df
    

def get_flashes(args):
    """
    Get flashes for given place

    args : dict
           params to be given in creating url
    
    return metadata, data (list)    
    """

    # Getting flashes is slow. Don't do it for winter times
    #d = datetime.fromtimestamp(args['endtime'])
    #if int(d.strftime('%m')) < 6 or int(d.strftime('%m')) > 8:
    #    logging.debug('Not thunder storm season, returning 0 for flashes...')
    #    return pd.DataFrame()
    
    url = "http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?param=peak_current&producer=flash&format=ascii&separator=;&starttime={starttime}&endtime={endtime}&latlon={latlon}:{maxdistance}".format(**args)

    #url = "http://data.fmi.fi/fmi-apikey/9fdf9977-5d8f-4a1f-9800-d80a007579c9/timeseries?param=time,peak_current&producer=flash&format=ascii&separator=;&starttime=2017-06-08T00:00:00&endtime=2017-06-08T18:00:00&latlon=57.1324,24.3574:30&timeformat=epoch&tz=local"
    
    logging.debug('Using url: {}'.format(url))

    rawdata = []
    try:
        with urllib.request.urlopen(url, timeout=600) as u:
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

    with urllib.request.urlopen(url, timeout=6000) as u:
        #rawdata = json.loads(u.read().decode("utf-8"))
        rawdata = u.read().decode("utf-8")
    logging.debug('Observations loaded')
    
    if len(rawdata) == 0:
        raise ValueError('No data for location {} and time {}'.format(args['latlon'], args['endtime']))

    obsf = StringIO(rawdata)    
    obs_df = pd.read_csv(obsf, sep=";", header=None)
    obs_df.rename(columns={1:'time'}, inplace=True)

    return obs_df

def process_timerange(starttime, endtime, params, producer, latlons, ids):

    """
    Process timerange
    """
    a = mlfdb.mlfdb(options.db_config_file)
    io = lib.io.IO()

    startstr = starttime.strftime('%Y-%m-%dT%H:%M:%S')
    endstr = endtime.strftime('%Y-%m-%dT%H:%M:%S')
    
    logging.info('Processing time {} - {}...'.format(startstr, endstr))

    # Get labels, features and filter labels that do not have features
    l_metadata, l_header, l_data = a.get_rows(options.dataset, rowtype='label', starttime=starttime, endtime=endtime)
    f_metadata, f_header, f_data = a.get_rows(options.dataset, rowtype='feature', starttime=starttime, endtime=endtime)    
    filt_metadata, _ = io.filter_labels(l_metadata, l_data, f_metadata, f_data, invert=True, uniq=True)

    latlons = set()
    for row in filt_metadata:
        latlons.add(str(row[3])+','+str(row[2]))
                    
    logging.info('There are {} (of total {}) lines to process...'.format(len(filt_metadata), len(l_metadata)))    

    if len(filt_metadata) == 0:
        return 0

    # Create url and get data
    apikey = '9fdf9977-5d8f-4a1f-9800-d80a007579c9'
    count = 0
    
    for latlon in latlons:

        label_metadata = io.filter_labels_by_latlon(filt_metadata, latlon)
        starttime, endtime = io.get_timerange(label_metadata)
        
        startstr = starttime.strftime('%Y-%m-%dT%H:%M:%S')
        endstr = endtime.strftime('%Y-%m-%dT%H:%M:%S')
        
        args = {
            'latlon': latlon,
            'params': ','.join(params),
            'starttime' : startstr,
            'endtime': endstr,
            'producer': options.producer,
            'apikey' : apikey,
            'maxdistance' : 100000 
        }
        
        flash_start = starttime - timedelta(hours=1)
        flash_args = {
            'starttime' : flash_start.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime' : endstr,
            'latlon' : latlon,
            'apikey' : apikey,
            'maxdistance' : 30 # Harmonie FlashMultiplicity value in km
        }

        prec_start = starttime - timedelta(hours=6)
        prec_args = {
            'producer' : options.producer,
            'starttime' : prec_start.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime' : endstr,
            'latlon' : latlon,
            'apikey' : apikey,
            'maxdistance' : 100000 
        }

        #try:
        obs_df = get_ground_obs(params, args)
        obs_df = io.find_best_station(obs_df)        
        metadata_df, data_df = io.filter_ground_obs(obs_df, label_metadata)
            
        flash_df = get_flashes(flash_args)
        data_df = io.filter_flashes(flash_df, data_df)
            
        prec_df = get_prec_sum(prec_args)
        prec_df = io.find_best_station(prec_df)
        data_df = io.filter_precipitation(prec_df, data_df)

        #except timeout:
        #    logging.error('Timeout while fetching data...')
        #    continue
        #except ValueError as e:
        #    logging.error(e)
        #    continue

        data = np.array(data_df.drop(columns=['time', 2])).astype(np.float)
        metadata = metadata_df.as_matrix()
        
        logging.debug('Dataset size: {}'.format(data.shape))
        logging.debug('Metadata size: {}'.format(np.array(metadata).shape))

        if len(data) > 0:
            # Save to database        
            header = params[2:] + ['count(flash:60:0)', 'max(sum_t(precipitation1h:180:0))', 'max(sum_t(precipitation1h:360:0))']
        
            logging.debug('Inserting new dataset to db...')
            count += a.add_rows('feature', header, data, metadata, options.dataset)

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

    a = mlfdb.mlfdb(options.db_config_file)
    io = lib.io.IO()    
    params = io.read_parameters('cnf/parameters.txt')

    if options.replace:
        logging.info('Removing old dataset...')
        a.remove_dataset(options.dataset, type='feature')

    # Get locations and create coordinate list for SmartMet
    latlons, ids = get_latlons(a, options.dataset)
    
    # Split time range to chunks and process them in paraller
    start = datetime.strptime(options.starttime, '%Y-%m-%d')
    end = datetime.strptime(options.endtime, '%Y-%m-%d')

    res = [pool.apply_async(process_timerange, args=(chunk[0], chunk[1], params, options.producer, latlons, ids)) for chunk in split_timerange(start, end, days=1, hours=0)]
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
                        default='opendata',
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
