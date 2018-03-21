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

    a = mlfdb.mlfdb()
    stations = get_stations(filename=options.filename)

    locations = []
    for name, latlon in stations.items():
        locations.append([name, latlon['lat'], latlon['lon']])
    
    a.add_point_locations(locations, check_for_duplicates=True)

        
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        help='Location filename. If not set, tries to find locations from digitrafic.')
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
