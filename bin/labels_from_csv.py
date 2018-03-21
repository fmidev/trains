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

    train_types = {'K': 0,
                   'L': 1,
                   'T': 2,
                   'M': 3}
    
    a = mlfdb.mlfdb()
    X = read_data(options.filename, delimiter=',', remove='"')    
    stations = get_stations(filename='data/stations.json')

    locations = []
    names = []
    for name, latlon in stations.items():
        locations.append([name, latlon['lat'], latlon['lon']])        
        names.append(name)

    ids = a.get_locations_by_name(names)    

    if options.replace:
        logging.info('Removing old dataset...')
        a.remove_dataset(options.dataset, type='label')
    
    header = ['late_minutes', 'total_late_minutes', 'train_type', 'train_count']
    data = []
    metadata = []
    
    for row in X:
        timestr = row[0]+'T'+row[1]
        t = datetime.strptime(timestr, "%Y-%m-%dT%H") + timedelta(hours=1)
        loc = row[3]
        train_type = train_types[row[4]]
        late_minutes = int(row[7])
        total_late_minutes = int(row[5])
        train_count = int(row[9])
        
        try:
            metadata.append([t, find_id(ids, loc)])
            data.append([late_minutes, total_late_minutes, train_type, train_count])
        except:
            logging.error('No location data for {}'.format(loc))
            continue

    start = 0
    end = 0
    batch_size = 100
    row_offset = 0
    while start < len(data)-1:
        if start + batch_size > len(data)-1:
            end = len(data)-1
        else:
            end = start + batch_size
            
        logging.info('Insert rows {}-{}/{}...'.format(start, end, len(data)-1))
        row_offset += a.add_rows('label', header, np.array(data[start:end]), metadata[start:end], dataset=options.dataset, row_offset=row_offset)        
        start = end
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default=None, help='Dataset path and filename')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--replace',
                        action='store_true',
                        help='If set, old rows from the dataset are removed')
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
