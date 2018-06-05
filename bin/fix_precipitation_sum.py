import sys, os
import argparse
import logging
import datetime as dt
from datetime import timedelta
import json
import itertools
import numpy as np
import pandas as pd
# import tensorflow as tf
from mlfdb import mlfdb
import urllib.request, json
from io import StringIO
from ml_feature_db.api.mlfdb import mlfdb as db
from lib import io as _io
from lib import viz as _viz

def get_prec_sum(args, i, total):
    """
    Get precipitation sum
    """
    url = "http://data.fmi.fi/fmi-apikey/{apikey}/timeseries?format=ascii&separator=;&producer={producer}&tz=local&timeformat=epoch&latlon={latlon}&timestep=60&starttime={starttime}&endtime={endtime}&param=stationname,time,place,lat,lon,precipitation1h&maxdistance={maxdistance}&numberofstations=5".format(**args)

    logging.info('Fetching precipitation data (location {}/{})...'.format(i, total))
    logging.debug('Using url: {}'.format(url))

    with urllib.request.urlopen(url, timeout=300) as u:
        rawdata = u.read().decode("utf-8")

    logging.info('Precipitation data loaded')

    if len(rawdata) == 0:
        return pd.DataFrame()

    obsf = StringIO(rawdata)
    obs_df = pd.read_csv(obsf, sep=";", header=None)
    obs_df.rename(columns={1:'time'}, inplace=True)
    return obs_df

def main():
    """
    """

    pd.set_option('display.max_rows', 6)
    apikey = '9fdf9977-5d8f-4a1f-9800-d80a007579c9'

    #a = mlfdb.mlfdb()
    a = db.mlfdb()
    io = _io.IO()
    viz = _viz.Viz()

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    param_names = ['max_precipitation1h', 'precipitation3h', 'precipitation6h']
    meta_columns = ['loc_id', 'time', 'lon', 'lat']
    params = ['time','stationname','precipitation1h']

    day_step = 10
    hour_step = 0

    start = starttime
    end = starttime + timedelta(days=day_step, hours=hour_step)
    if end > endtime: end = endtime

    while end <= endtime:
        logging.info('Processing time range {} - {}'.format(start.strftime('%Y-%m-%d %H:%M'), end.strftime('%Y-%m-%d %H:%M')))

        try:
            data = a.get_rows(options.dataset,
                              starttime=start,
                              endtime=end,
                              rowtype='feature',
                              return_type='pandas',
                              parameters=param_names)

        except ValueError as e:
            print(e)
            start = end
            end = start + timedelta(days=day_step, hours=hour_step)
            continue

        if len(data) == 0:
            continue

        #print(data)
        data.sort_values(by=['loc_id', 'time'], inplace=True)

        locs = data.loc_id.unique()

        full_data = None

        i = 0
        for loc in locs:
            timerange = data.loc[data['loc_id'] == loc]
            loc_starttime = dt.datetime.fromtimestamp(timerange.iloc[0].time)
            loc_endtime = dt.datetime.fromtimestamp(timerange.iloc[-1].time)
            latlon = str(timerange.iloc[0].lat)+','+str(timerange.iloc[0].lon)
            query_starttime = loc_starttime - timedelta(hours=6)

            args = {
            'latlon': latlon,
            'params': ','.join(params),
            'starttime' : query_starttime.strftime('%Y-%m-%dT%H:%M:%S'),
            'endtime': loc_endtime.strftime('%Y-%m-%dT%H:%M:%S'),
            'producer': 'fmi',
            'apikey' : apikey,
            'maxdistance' : 100000
            }

            prec_data = io.find_best_station(get_prec_sum(args, (i+1), len(locs)))
            prec_data = io.filter_precipitation(prec_data, prec_column=5)

            prec_data.loc[:,'loc_id'] = timerange.iloc[0].loc_id
            prec_data.rename(index=str,
                             columns={5: 'max_precipitation1h',
                                      '3hsum': 'precipitation3h',
                                      '6hsum': 'precipitation6h'},
                            inplace=True)
            prec_data.drop(columns=[2,3,4], inplace=True)
            prec_data.fillna(-99, inplace=True)

            if len(data) > 0:
                if full_data is None:
                    full_data = prec_data
                else:
                    full_data = full_data.append(prec_data)
            i += 1

        logging.info('Updating db ({} rows)...'.format(len(full_data)))
        count = a.update_rows_df(df=full_data,
                                 _type='feature',
                                 dataset=options.dataset,
                                 columns=param_names
                                 )
        logging.info('{} rows updated'.format(count))

        start = end # - timedelta(hours=6)
        end = start + timedelta(days=day_step, hours=hour_step)

    logging.info('Finished')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, help='End time of the classification data interval')
    parser.add_argument('--output_path', type=str, default=None, help='Output save path')
    parser.add_argument('--dataset', type=str, default=None, help='Source dataset name')
    parser.add_argument('--db_config_file',
                        type=str,
                        default=None,
                        help='GS address for db config file (if none, ~/.mlfdbconfig is used)')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    options = parser.parse_args()

    if options.output_path is not None and not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
