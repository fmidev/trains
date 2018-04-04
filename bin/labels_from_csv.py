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
import multiprocessing
from itertools import repeat

from mlfdb import mlfdb
from lib import io as oi

def process_rows(X, header, ids, dataset, io, row_prefix=0):
    """
    Process rows in file
    """

    logging.info('Procssing chunk with length {}...'.format(len(X)))

    a = mlfdb.mlfdb(config_filename=options.db_config_file)
    train_types = {'K': 0,
                   'L': 1,
                   'T': 2,
                   'M': 3}

    data = []
    metadata = []
    errors = set()
    batch_size = 1000
    inserted_count = 0
    
    for row in X:
        try:
            timestr = row[0]+'T'+row[1]
            t = datetime.strptime(timestr, "%Y-%m-%dT%H") + timedelta(hours=1)
            loc = row[3]
            train_type = train_types[row[4]]
            late_minutes = int(row[7])
            total_late_minutes = int(row[5])
            train_count = int(row[9])
        except Exception as e:
            logging.error(e)
            continue
        
        try:
            metadata.append([t, io.find_id(ids, loc)])
            data.append([late_minutes, total_late_minutes, train_type, train_count])
        except Exception as e:
            errors.add(loc)
            continue

        # Save data in batches
        if len(data) >= batch_size:
            inserted_count += a.add_rows('label', header, np.array(data), metadata, dataset=dataset)
            data = []
            metadata = []
            if inserted_count%5000 == 0:
                logging.info('{} rows inserted...'.format(inserted_count))

    if len(errors) > 0:
        logging.error('Coordinates not found for locations: {}'.format(','.join(errors)))
    
    return inserted_count
    

def process_file(filename, dataset, ids, a, io, job_count=1, row_prefix=0):
    """
    Process file content and save it to database
    """

    logging.info('Processing file {}...'.format(filename))

    # Initialize process pool
    if job_count < 0:
        job_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(job_count)
    
    # Get data
    filename = io.get_file_to_process(filename)
    X = io.read_data(filename, delimiter=',', remove='"')
    header = ['late_minutes', 'total_late_minutes', 'train_type', 'train_count']
    
    # Split data to chunks
    chunks = list(io.chunks(X, round(len(X)/job_count)))
    res = [pool.apply_async(process_rows, args=(chunk, header, ids, dataset, io)) for chunk in chunks]
    res = [p.get() for p in res]
    
    logging.info('Inserted {} rows into db'.format(sum(res)))


    
            
def main():
    """
    Put labels from csv file to db
    """
    
    io = oi.IO(gs_bucket=options.gs_bucket)
    a = mlfdb.mlfdb(config_filename=options.db_config_file)

    # Remove old dataset
    if options.replace:
        logging.info('Removing old dataset...')
        a.remove_dataset(options.dataset, type='label')

    # Get stations
    stations = io.get_train_stations(filename='data/stations.json')

    locations = []
    names = []
    for name, latlon in stations.items():
        names.append(name)

    ids = a.get_locations_by_name(names)    

    # Process files
    if options.filename is not None:
        files = [options.filename]
    else:
        files = io.get_files_to_process('data', 'csv')

    logging.info('Processing files: {}'.format(','.join(files)))

    count = 0
    for file in files:
        process_file(options.filename, options.dataset,
                     ids, a, io, job_count=options.job_count)
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        help='Dataset path and filename')
    parser.add_argument('--job_count',
                        type=int,
                        default=1,
                        help='Job count for processing. Use -1 for using all cores')
    parser.add_argument('--gs_bucket',
                        type=str,
                        default=None,
                        help='Google Cloud bucket name to use for downloading data')
    parser.add_argument('--path',
                        type=str,
                        default=None,
                        help='Path to find files to process. Use filename or bath, not both. If gs_bucket is given, Google Cloud bucket is used.')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='Dataset name')
    parser.add_argument('--replace',
                        action='store_true',
                        help='If set, old rows from the dataset are removed')
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
