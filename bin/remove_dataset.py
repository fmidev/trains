#!/usr/bin/python
import sys
import argparse
import logging
from datetime import datetime, timedelta
import json
import itertools
import numpy as np
import urllib.request, json
import requests
import codecs

from mlfdb import mlfdb

def read_data(filename, data_type=None, delimiter=';', skip_cols=0, skip_rows=1, remove=None):
    """ 
    Read data from csv file
    """
    X = []
    with open(filename, encoding='utf-8') as f:
        lines = f.read().splitlines()

    for line in lines[skip_rows:]:
        l = line.split(delimiter)[skip_cols:]
        if remove is not None:
            nl = []
            for e in l:
                nl.append(e.replace(remove, ''))
            l = nl
        if data_type is not None:
            l = list(map(data_type, l))
        X.append(l)

    return X

def get_stations(filename=None):
    """
    Get railway stations from digitrafic
    """
    if filename is None:
        url = "https://rata.digitraffic.fi/api/v1/metadata/stations"

        with urllib.request.urlopen(url) as u:
            data = json.loads(u.read().decode("utf-8"))
    else:
        with open(filename) as f:
            data = json.load(f)

    stations = dict()
       
    for s in data:
        latlon = {'lat': s['latitude'], 'lon': s['longitude']}
        stations[s['stationShortCode'].encode('utf-8').decode()] = latlon

    return stations

def find_id(locations, name):
    """
    Find id from (id, name) tuple list
    """
    for loc in locations:
        if name == loc[1]:
            return loc[0]

    return None
    
def main():
    """
    Put labels from csv file to db
    """

    a = mlfdb.mlfdb(config_filename=options.db_config_file)

    if options.labels:
        logging.info('Removing labels from {}...'.format(options.dataset))
        a.remove_dataset(options.dataset, type='label')
    if options.features:
        logging.info('Removing features from {}...'.format(options.dataset))
        a.remove_dataset(options.dataset, type='feature')
    
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--labels',
                        action='store_true',
                        help='If set, labels are removed')
    parser.add_argument('--features',
                        action='store_true',
                        help='If set, features are removed')
    parser.add_argument('--db_config_file',
                        type=str,
                        default=None,
                        help='GS address for db config file (if none, ~/.mlfdbconfig is used)')
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
