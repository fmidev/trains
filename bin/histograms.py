import sys,os
import matplotlib as mlp
mlp.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import argparse, logging, json
import datetime as dt
from datetime import timedelta

import itertools
from collections import OrderedDict

import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.python.estimator.export import export
from tensorflow.python.framework import constant_op
from tensorflow.contrib import predictor

from sklearn import metrics

import lib.io as _io
import lib.viz as _viz
import lib.bqhandler as _bq
import lib.config as _config

logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging.INFO)

#path = 'labels_vis_1_1/histograms'
path = 'results/histograms/dual_trains_data_2010_2019_winters_3_test'

if not os.path.exists(path):
    os.makedirs(path)

class Options():
    pass

options = Options()
options.starttime = '2010-01-01'
options.endtime = '2018-01-01'
options.config_filename = 'cnf/rf.ini'
options.config_name = 'all_params_1'
options.stations_file = 'cnf/stations.json'
options.stations = None #'PSL,OL,TPE,OV,PM,II,KEM,HKI'
options.gs_bucket = 'trains-data'

_config.read(options)

bq = _bq.BQHandler()
io = _io.IO(gs_bucket=options.gs_bucket)
viz = _viz.Viz()

starttime, endtime = io.get_dates(options)

# Get params
all_param_names = options.label_params + options.feature_params + options.meta_params
aggs = io.get_aggs_from_param_names(options.feature_params)

print('Loading stations from {}...'.format(options.stations_file))
stationList = io.get_train_stations(options.stations_file)
if options.stations is not None:
    stations = options.stations.split(',')
else:
    stations = stationList.keys()

print('Loading data...')
all_data = bq.get_rows(starttime,
                       endtime,
                       loc_col='trainstation',
                       project=options.project,
                       dataset='trains_data',
                       table='dual_trains_data_2010_2019_winters_3_test',
                       parameters=all_param_names,
                       locations=stations)

print('Filtering data...')
all_data = io.filter_train_type(labels_df=all_data,
                            train_types=['K','L'],
                            sum_types=True,
                            train_type_column='train_type',
                            location_column='trainstation',
                            time_column='time',
                            sum_columns=['delay'],
                            aggs=aggs)

all_data.sort_values(by=['time', 'trainstation'], inplace=True)
print('Data contain {} rows...'.format(len(all_data)))

filt_data = all_data.replace(-99, np.nan)
delayed_data = filt_data[filt_data.loc[:,'delay'] > 50]

# All in one
print('Plotting histograms all in one...')
d = {'A': filt_data, 'B': delayed_data}
comp_data = pd.concat(d.values(), axis=1, keys=d.keys())

axes = comp_data['A'].hist(figsize=(20,20), bins=50, density=True, label='All delays')
axes = comp_data['B'].hist(bins=50, ax = axes, density=True, alpha=0.5, label='Over 50 minutes delays')
plt.legend()

filename = path+'/histograms_all_params_compare.png'
viz._save(plt, filename)
io._upload_to_bucket(filename, filename)

# One by one
print('Plotting histograms one by one...')
for param in list(filt_data)[2:]:
    print('{}...'.format(param))
    fig, ax = plt.subplots(figsize=(20,10))
    axes = filt_data.loc[:,param].hist(figsize=(20,10), bins=50, density=True, alpha=0.5, label='All delays', ax=ax)
    axes = delayed_data.loc[:,param].hist(figsize=(20,10), bins=50, density=True, alpha=0.5, label='Delays over 50 minutes', ax=ax)
    plt.title('Histogram of {}'.format(param))
    plt.legend()
    filename = path+'/histograms_{}_compare.png'.format(param)
    viz._save(plt, filename)
    io._upload_to_bucket(filename, filename)

# Record count
print('Plotting record count...')
time_data = filt_data.loc[:,['time', 'delay']]
time_data.loc[:,'time'] = pd.to_datetime(time_data.loc[:,'time'], format='%Y-%m-%d')
time_data.set_index('time', inplace=True)
fig, ax = plt.subplots(figsize=(20,10))
time_data.resample('24H').count().plot(title='Amount of daily records', figsize=(20,10), ax=ax)
ax.yaxis.set_label('Record count')
fig.autofmt_xdate()
filename = path+'/record_count.png'
viz._save(plt, filename)
io._upload_to_bucket(filename, filename)
