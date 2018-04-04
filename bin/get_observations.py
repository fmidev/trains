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

from mlfdb import mlfdb
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

def process_timerange(starttime, endtime, params, producer, ids):
    """
    Process timerange
    """
    a = mlfdb.mlfdb(options.db_config_file)
    io = lib.io.IO()
    
    logging.info('Processing time {} - {}...'.format(starttime.strftime('%Y-%m-%d %H:%M'), endtime.strftime('%Y-%m-%d %H:%M')))

    l_metadata, l_header, l_data = a.get_rows(options.dataset, rowtype='label', starttime=starttime, endtime=endtime)
    f_metadata, f_header, f_data = a.get_rows(options.dataset, rowtype='feature', starttime=starttime, endtime=endtime)    
    f_metadata, _ = io.filter_labels(l_metadata, l_data, f_metadata, f_data, invert=True)

    handled = set()
    count = 0
    for row in f_metadata:
        count += 1
        time = int(row[0])
        latlon = '{},{}'.format(row[3], row[2])

        timepoint = (time, latlon)
        if timepoint in handled:
            continue
        else:
            handled.add(timepoint)
        
        # Create url and get data
        url = 'http://data.fmi.fi/fmi-apikey/9fdf9977-5d8f-4a1f-9800-d80a007579c9/timeseries?format=json&producer={producer}&tz=local&timeformat=epoch&latlon={latlon}&timesteps=1&starttime={endtime}&param={params},stationname&attributes=stationname&numberofstations=3'.format(latlon=latlon, params=','.join(params), endtime=time, producer=options.producer)

        logging.debug('Loading data from SmartMet Server...')
        logging.debug('Using url: {}'.format(url))

        try:
            with urllib.request.urlopen(url, timeout=6000) as u:
                rawdata = json.loads(u.read().decode("utf-8"))
        except timeout as e:
            logging.error(e)
            logging.error(url)
            continue

        if len(rawdata) == 0:
            logging.error('No data for location {} and time {}'.format(latlon, datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')))
            continue
        
        # Parse data to numpy array
        logging.debug('Parsing data to np array...')
        data, metadata, row = [], [], []        

        # Go through stations
        for station,values in rawdata.items():
            # if station is empty, continue to next station
            if len(values) == 0:
                continue

            # Go through rows and parameters
            for el in values:
                for param in params[2:]:
                    if el[param] is None:
                        row.append(-99)
                    else:
                        row.append(el[param])

            # if row is not empty don't continue to next stations
            if len(row) > 0:
                metadata.append([int(el['time']), ids[el['place']], el['lon'], el['lat']])
                data.append(row)
                break
            
        data = np.array(data).astype(np.float)
        logging.debug('Dataset size: {}'.format(data.shape))
        logging.debug('Metadata size: {}'.format(np.array(metadata).shape))
                
        if count%10 == 0:
            logging.info('Handled {}/{} rows...'.format(count, len(f_metadata)))

        # Save to database        
        header = params[2:]
        data = np.array(data)
    
        logging.debug('Inserting new dataset to db...')
        a.add_rows('feature', header, data, metadata, options.dataset)

    return len(data)

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
    latlons = []
    ids = dict()
    if options.locations is not None:
        locations = options.locations.split(';')
        for loc in locations:
            id, latlon = loc.split(':')
            latlons.append(latlon)
            ids[latlon] = id
    else:
        locations = a.get_locations_by_dataset(options.dataset)
        for loc in locations:
            latlon = str(loc[3])+','+str(loc[2])
            latlons.append(latlon)
            ids[latlon] = loc[0]


    # Split time range to chunks and process them in paraller
    start = datetime.strptime(options.starttime, '%Y-%m-%d')
    end = datetime.strptime(options.endtime, '%Y-%m-%d')

    res = [pool.apply_async(process_timerange, args=(chunk[0], chunk[1], params, options.producer, ids)) for chunk in split_timerange(start, end, days=0, hours=6)]
    res = [p.get() for p in res]

    logging.info('Added {} rows observations to db.'.format(sum(res)))
    
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
    parser.add_argument('--locations',
                        type=str,
                        default=None,
                        help='Locations in format [id:lat,lon;id:lat,lon...], default None')
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
