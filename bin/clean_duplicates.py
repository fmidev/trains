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

def main():
    """
    Clean duplicate entries
    """
    
    # Initialize process pool
    a = mlfdb.mlfdb(options.db_config_file)
    io = lib.io.IO()    

    removed = a.clean_duplicate_rows(options.dataset, options.type, options.count)

    logging.info('Removed {} {}s from dataset {}...'.format(removed, options.type, options.dataset))
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        help='Name of dataset bind to locations')
    parser.add_argument('--type',
                        type=str,
                        default='feature',
                        help='Rowtype (label/feature)')
    parser.add_argument('--count',
                        type=int,
                        default=29,
                        help='How many features one row should contain')
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
