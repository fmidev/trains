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
    url = "http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?format=json&producer={producer}&tz=local&timeformat=epoch&latlon={latlon}&timesteps=6&timestep=60&starttime={endtime}&param=precipitation1h,stationname&attributes=stationname&numberofstations=3&maxdistance={maxdistance}".format(**args)

    logging.debug('Using url: {}'.format(url))
    
    with urllib.request.urlopen(url, timeout=600) as u:
        rawdata = json.loads(u.read().decode("utf-8"))        

    logging.debug('Precipitation data loaded')
    
    if len(rawdata) == 0:
        return [-99, -99]

    # Go through stations
    prec_sums_6h, prec_sums_3h = [], []
    for station,values in rawdata.items():
        # if station is empty, continue to next station
        if len(values) == 0:
            continue
        
        # Go through times
        prec_sum_6h = 0
        prec_sum_3h = 0
        i = 0
        for step in values:
            i += 1                
            try:
                prec = float(step['precipitation1h'])
                if prec >= 0:
                    prec_sum_6h += prec
                    if i > 3:
                        prec_sum_3h += prec
                    found = True
            except:
                break
        prec_sums_6h.append(prec_sum_6h)
        prec_sums_3h.append(prec_sum_3h)

    if not found:
        return [-99, -99]
        
    return [max(prec_sums_3h), max(prec_sums_6h)]

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
        return 0
    
    url = "http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?param=peak_current&producer=flash&format=json&starttime={starttime}&endtime={endtime}&latlon={latlon}:{maxdistance}".format(**args)

    logging.debug('Using url: {}'.format(url))
    
    with urllib.request.urlopen(url, timeout=600) as u:
        count = len(json.loads(u.read().decode("utf-8")))

    logging.debug('Flash data loaded')
    return count
        
def get_ground_obs(params, ids, args):
    """
    Get ground observations which can be fetched directly from
    timeseries
    
    params : list
             list of params
    ids : dict
          ids
    args : dict
           params to be given in creating url
    
    return metadata, data (list)
    """

    url = 'http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?format=json&producer={producer}&tz=local&timeformat=epoch&latlon={latlon}&timesteps=1&starttime={endtime}&param={params},stationname&attributes=stationname&numberofstations=3&maxdistance={maxdistance}'.format(**args)
    
    logging.debug('Loading data from SmartMet Server...')
    logging.debug('Using url: {}'.format(url))

    with urllib.request.urlopen(url, timeout=6000) as u:
        rawdata = json.loads(u.read().decode("utf-8"))        

    if len(rawdata) == 0:
        logging.error('No data for location {} and time {}'.format(args['latlon'], datetime.fromtimestamp(args['endtime']).strftime('%Y-%m-%d %H:%M:%S')))
        raise ValueError('No data found')
    
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
        
    return metadata, data


def process_timerange(starttime, endtime, params, producer, ids):

    """
    Process timerange
    """
    a = mlfdb.mlfdb(options.db_config_file)
    io = lib.io.IO()
    
    logging.info('Processing time {} - {}...'.format(starttime.strftime('%Y-%m-%d %H:%M:%S'), endtime.strftime('%Y-%m-%d %H:%M:%S')))

    l_metadata, l_header, l_data = a.get_rows(options.dataset, rowtype='label', starttime=starttime, endtime=endtime)
    f_metadata, f_header, f_data = a.get_rows(options.dataset, rowtype='feature', starttime=starttime, endtime=endtime)    
    filt_metadata, _ = io.filter_labels(l_metadata, l_data, f_metadata, f_data, invert=True, uniq=True)

    # Checking handled rows should not be needed but is for safety
    handled = set() 
    count = 0
    for row in filt_metadata:
        count += 1
        time = int(row[0])
        latlon = '{},{}'.format(row[3], row[2])

        timepoint = (time, latlon)
        if timepoint in handled:
            continue
        else:
            handled.add(timepoint)
            log_process(count, len(filt_metadata))
        
        # Create url and get data
        apikey = '9fdf9977-5d8f-4a1f-9800-d80a007579c9'

        args = {
            'latlon': latlon,
            'params': ','.join(params),
            'endtime': time,
            'producer': options.producer,
            'apikey' : apikey,
            'maxdistance' : 50000 # default
        }
        flash_args = {
            'starttime' : (time - 3600),
            'endtime' : time,
            'latlon' : latlon,
            'apikey' : apikey,
            'maxdistance' : 30000 # Harmonie FlashMultiplicity value
        }
        prec_args = {
            'producer' : options.producer,
            'endtime' : time,
            'latlon' : latlon,
            'apikey' : apikey,
            'maxdistance' : 50000 # Default
        }

        try:
            metadata, data = get_ground_obs(params, ids, args)
            data[0].append(get_flashes(flash_args))
            data[0] += get_prec_sum(prec_args)
        except timeout:
            logging.error('Timeout while fetching data...')
            continue
        except ValueError as e:
            logging.error(e)
            continue
        except Exception as e:
            logging.error(e)
            continue
        
        data = np.array(data).astype(np.float)
        
        logging.debug('Dataset size: {}'.format(data.shape))
        logging.debug('Metadata size: {}'.format(np.array(metadata).shape))
        log_process(count, len(filt_metadata))
        
        # Save to database        
        header = params[2:] + ['count(flash:60:0)', 'max(sum_t(precipitation1h:180:0))', 'max(sum_t(precipitation1h:360:0))']
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
