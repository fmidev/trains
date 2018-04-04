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

import pyspark

from mlfdb import mlfdb
from lib import io as oi


def process_file(filename, dataset, ids, a, io, row_prefix=0):
    """
    Process file content and save it to database
    """

    logging.info('Processing file {}...'.format(filename))
    
    train_types = {'K': 0,
                   'L': 1,
                   'T': 2,
                   'M': 3}
    
    # Get data
    filename = io.get_file_to_process(filename)
    X = io.read_data(filename, delimiter=',', remove='"')
    header = ['late_minutes', 'total_late_minutes', 'train_type', 'train_count']
    data = []
    metadata = []
    errors = set()
    
    for row in X:
        timestr = row[0]+'T'+row[1]
        t = datetime.strptime(timestr, "%Y-%m-%dT%H") + timedelta(hours=1)
        loc = row[3]
        train_type = train_types[row[4]]
        late_minutes = int(row[7])
        total_late_minutes = int(row[5])
        train_count = int(row[9])
        
        try:
            metadata.append([t, io.find_id(ids, loc)])
            data.append([late_minutes, total_late_minutes, train_type, train_count])
        except Exception as e:
            errors.add(loc)
            continue

    logging.error('Location not found for locations: {}'.format(','.join(errors)))
    
    # Save data
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
        row_offset += a.add_rows('label', header, np.array(data[start:end]), metadata[start:end], dataset=dataset) #, row_offset=row_offset)        
        start = end

    return end
    
def main():
    """
    Put labels from csv file to db
    """
    
    io = oi.IO(gs_bucket=options.gs_bucket)
    a = mlfdb.mlfdb(config_filename=options.db_config_file)

    # sc = pyspark.SparkContext("local")
    # sc = pyspark.SparkContext('spark://q2-m.c.trains-197305.internal:7077')
    # SparkSession.builder.config(conf=SparkConf())
    sc = spark.sparkContext
    #conf = pyspark.SparkConf()
    #conf.setMaster('yarn')
    #sc = pyspark.SparkContext(conf)

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
    count = sc.accumulator(0)
    if options.filename is not None:
        files = [options.filename]
    else:
        files = io.get_files_to_process('data', 'csv')

    logging.info('Processing files: {}'.format(','.join(files)))
       
    sc.parallelize(files).foreach(lambda filename: count.add(process_file(filename, options.dataset, ids, a, io)))
    
    logging.info('Added {} samples to db'.format(count.value)) 

    #process_file(options.filename, options.dataset, ids, a, io)
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        help='Dataset path and filename')
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
