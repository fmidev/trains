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
        
def get_forecasts(args):
    """
    Get forecasts
    
    args : dict
           params to be given in creating url
    
    return pandas df
    """

    url = 'http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?format=ascii&separator=;&producer={producer}&tz=local&timeformat=epoch&endtime=data&latlons={latlons}&param={params}#{names}'.format(**args)
    
    logging.debug('Loading data from SmartMet Server...')
    logging.debug('Using url: {}'.format(url))

    with urllib.request.urlopen(url, timeout=6000) as u:
        rawdata = args['params'].replace(',',';')+'\n'+u.read().decode("utf-8")
    logging.debug('Data loaded')
    
    obsf = StringIO(rawdata)    
    df = pd.read_csv(obsf, sep=";")    
    
    return df
    

    
    # flash_start = starttime - timedelta(hours=1)
    # flash_args = {
    #     'starttime' : flash_start.strftime('%Y-%m-%d %H:%M:%S'),
    #     'endtime' : endstr,
    #     'latlon' : latlon,
    #     'apikey' : apikey,
    #     'maxdistance' : 30000 # Harmonie FlashMultiplicity value
    # }
    prec_args = {
        'producer' : options.producer,
        'starttime' : startstr,
        'endtime' : endstr,
        'latlon' : ','.join(latlons),
        'apikey' : apikey,
        'maxdistance' : 50000 # Default
    }

    try:
        obs_df = get_ground_obs(params, args)
        metadata, data = io.filter_ground_obs(obs_df, filt_metadata)
        print(metadata)
        #data[0].append(get_flashes(flash_args))
        #data[0] += get_prec_sum(prec_args)
    except timeout:
        logging.error('Timeout while fetching data...')
    except ValueError as e:
        logging.error(e)
        return 0    
                
    data = np.array(data).astype(np.float)
    
    logging.debug('Dataset size: {}'.format(data.shape))
    logging.debug('Metadata size: {}'.format(np.array(metadata).shape))
        
    # Save to database        
    header = params[2:] #+ ['count(flash:60:0)', 'max(sum_t(precipitation1h:180:0))', 'max(sum_t(precipitation1h:360:0))']
    
    logging.debug('Inserting new dataset to db...')
    count = a.add_rows('feature', header, data, metadata, options.dataset)

    return count

def main():
    """
    Get forecasted delay for every station
    """

    io = lib.io.IO()
    params,names = io.read_parameters(options.parameters_filename)
    stations = io.get_train_stations(filename=options.stations_filename)

    max_count = 1
    latlons = []
    names = []
    for name, station in stations.items():
        if len(latlons) > max_count:
            break
        latlons.append(str(station['lat'])+','+str(station['lon']))
        names.append(name)
    
    print(params)
    logging.info('Getting delay forecast for {} locations...'.format(len(stations)))

    # Create url and get data
    apikey = '9fdf9977-5d8f-4a1f-9800-d80a007579c9'

    args = {
        'latlons': ','.join(latlons),
        'names' : ','.join(names),
        'params': ','.join(params),
        'producer': 'harmonie_skandinavia_pinta',
        'apikey' : apikey
    }
    
    data = get_forecasts(args)

    data = io._calc_prec_sums(data, prec_column='PrecipitationInstantTotal').fillna(-99)

    serving = io.df_to_serving_json(data)
    result = io.predict_json('trains', 'trains_lr', serving, version='tiny_subset_2')
                          
    print(data)
    print(serving_json)
    
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
