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
import requests
import codecs
import multiprocessing

from mlfdb import mlfdb
import lib.io

def pointtime_in_metadata(metadata, location_id, time):
    for row in metadata:
        if int(row[0]) == int(time) and int(row[1]) == int(location_id):
            return True
    return False

def split_timerange(starttime, endtime, days=1, hours=0):
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

    return list of tuples [(starttime:Datetime, endtime:Datetime)]
    """
    chunks = []
    start = starttime
    end = start + timedelta(days=days, hours=hours)
    while end <= endtime:        
        chunks.append((start, end))
        start = end
        end = start + timedelta(days=days, hours=hours)
    chunks.append((start, end))
    return chunks

def process_timerange(latlons, starttime, endtime, timestep, params, producer, ids):
    """
    Process timerange
    """

    startstr = starttime.strftime('%Y-%m-%dT%H:%M:%S')
    endstr = endtime.strftime('%Y-%m-%dT%H:%M:%S')
    logging.info('Processing time {} - {}...'.format(startstr, endstr))

    a = mlfdb.mlfdb(options.db_config_file)
    # Create url and get data
    url = 'http://data.fmi.fi/fmi-apikey/9fdf9977-5d8f-4a1f-9800-d80a007579c9/timeseries?format=json&producer={producer}&tz=local&timeformat=epoch&latlons={latlons}&timestep={timestep}&starttime={starttime}&endtime={endtime}&param={params}'.format(latlons=','.join(latlons), timestep=options.timestep, params=','.join(params), starttime=startstr, endtime=endstr, producer=options.producer)

    logging.info('Loading data from SmartMet Server...')
    logging.debug('Using url: {}'.format(url))

    with urllib.request.urlopen(url, timeout=6000) as u:
        data = json.loads(u.read().decode("utf-8"))
        
    # Parse data to numpy array
    logging.debug('Parsing data to np array...')
    result = []
    for el in data:
        row = []
        for param in params:
            if el[param] is None:
                row.append(-99)
            else:
                row.append(el[param])
        result.append(row)
    result = np.array(result)
    places = result[:,1]
    result = np.delete(result, 1, 1).astype(np.float)
    logging.debug('Dataset size: {}'.format(result.shape))
    
    # Result by time
    logging.debug('Arranging by time...')
    result = result[result[:,0].argsort()]

    # Go through data, remove unnecessary rows and convert coordinates
    # back to location ids
    logging.info('Decimating unnecessary data...')
    logging.debug('Reading labels data...')
    labels_metadata, _, __ = a.get_rows(options.dataset, rowtype='label')
    
    metadata = []
    data = []
    removed = 0
    count = 0
    
    # Masks don't work because result and labels are not necessary in the same order
    #lm = np.array(labels_metadat)
    #mask = np.ones_like(result)*(places[:]==result[:,1])
    #result_time = np.ma.MaskedArray(result, mask)
    #logging.debug("Length of dataset after masking places: {}".format(len(result_time)))

    #mask = np.ones_like(result_time)*(lm[:,0]==result_time[:,0])
    #logging.debug("Length of dataset after masking times: {}".format(len(np.ma.MaskedArray(result_time, mask))))

    for row in result:
        if pointtime_in_metadata(labels_metadata, ids[places[count]], row[0]):
            metadata.append([int(row[0]), ids[places[count]]])
            data.append(row[1:])
        else:
            removed += 1
        
        count += 1
        if count%500 == 0:
            logging.info('Handled {}/{} rows ({} rows removed)...'.format(count, len(result), removed))
        
    logging.info('Removed {} rows from data'.format(removed))

    # Save to database        
    header = params[2:]
    data = np.array(data)
    
    if options.replace:
        logging.info('Removing old dataset...')
        a.remove_dataset(options.dataset, type='feature')

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

    io = lib.io.IO()    
    params = io.read_parameters('cnf/parameters.txt')
    
    # Get locations and create coordinate list for SmartMet
    a = mlfdb.mlfdb(options.db_config_file)
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

    res = [pool.apply_async(process_timerange, args=(latlons, chunk[0], chunk[1], options.timestep, params, options.producer, ids)) for chunk in split_timerange(start, end, days=0, hours=6)]
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
