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

# from mlfdb import mlfdb
from lib import io as _io
#import lib.manipulator
from lib import dbhandler as _db
from lib import viz as _viz

def main():
    """
    Get forecasted delay for every station
    """

    io = _io.IO()
    viz = _viz.Viz()

    params, _ = io.read_parameters(options.parameters_filename)
    model = io.load_scikit_model(options.model_file)

    print(params)
    print(len(params))
    print(len(model.feature_importances_))
    fname = 'results/manual/rfc_feature_importance.png'
    viz.rfc_feature_importance(model.feature_importances_,
                               fname,
                               feature_names=params[2:]+['precipitation3h','precipitation6h'], 
                               fontsize=18)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--stations_filename',
                        type=str,
                        default='cnf/stations.json',
                        help='Stations file name')
    parser.add_argument('--model_file',
                        type=str,
                        default=None,
                        help='Path and filename of SciKit model file.')
    parser.add_argument('--parameters_filename',
                        type=str,
                        default='cnf/forecast_parameters_shorten.txt',
                        help='Parameters file name')
    parser.add_argument('--prec3h',
                        type=int,
                        default=1,
                        help='Calculate precipitation 3 hour')
    parser.add_argument('--prec6h',
                        type=int,
                        default=1,
                        help='Calculate precipitation 6 hour')
    parser.add_argument('--flashcount',
                        type=int,
                        default=1,
                        help='Calculate flashcount')
    parser.add_argument('--dev',
                        type=int,
                        default=0,
                        help='Set 1 for development mode')
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
