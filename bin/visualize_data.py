import sys
import os
import argparse
import logging
import datetime as dt
import json
import itertools
import numpy as np
import pandas as pd
from sklearn import preprocessing

from lib import bqhandler as _bq
from lib import io as _io
from lib import viz as _viz

from multiprocessing import Pool, cpu_count

statlist = ['delay', 'total delay']
train_types = {'Intercity': 0,
               'Commuter': 1,
               'Cargo': 2,
               'Other': 3}

#a = _bq.BQHandler()
#io = _io.IO()
#viz = _viz.Viz()


def heatmap_day(l_data, passangers, day, locs):
    """
    Heatmap stations against day of the year

    l_data : DataFrame
            all labels data
    passangers : DataFrame
                passanger trains label data
    year : int
           year to handle
    """

    def heatmap_series(s, locs, order, filename):
        s = s.unstack(level=0)
        s.columns= s.columns.droplevel(0)
        s = s[order]
        s.set_index(s.index.hour, inplace=True)
        s = s.groupby(s.index)[s.columns].mean()
        viz.heatmap_train_station_delays(s, locs, filename, label='Hour')

    dstr = day.strftime('%Y-%m-%d')
    logging.info('Processing day: {}'.format(dstr))

    l_data = l_data.loc[dstr]
    passangers = passangers[dstr]

    order = l_data.sort_values(by=['lat', 'time'])['trainstation'].unique()
    pass_order = passangers.sort_values(by=['lat', 'time'])['trainstation'].unique()

    if len(l_data) == 0:
        return

    # All trains, late minutes
    filename = options.save_path+'/detailed_heatmap/late_minutes_over_day_all_{}.png'.format(dstr)

    df = l_data.groupby(['trainstation', l_data.index])['delay'].mean().to_frame()
    heatmap_series(df, locs, order, filename)

    # All trains, total_delay
    filename = options.save_path+'/detailed_heatmap/total_late_minutes_over_day_all_{}.png'.format(dstr)
    s = l_data.groupby(['trainstation', l_data.index])['total_delay'].mean().to_frame()
    heatmap_series(s, locs, order, filename)

    # Passanger trains, late minutes
    filename = options.save_path+'/detailed_heatmap/late_minutes_over_day_passanger_{}.png'.format(dstr)

    s = passangers.groupby(['trainstation', passangers.index])['delay'].mean().to_frame()
    heatmap_series(s, locs, pass_order, filename)

    # Passanger trains, total_delay
    filename = options.save_path+'/detailed_heatmap/total_late_minutes_over_day_passanger_{}.png'.format(dstr)

    s = passangers.groupby(['trainstation', passangers.index])['total_delay'].mean().to_frame()
    heatmap_series(s, locs, pass_order, filename)

    # for different train_types
    for name,t in train_types.items():

        df = l_data[l_data.loc[:,'train_type'].isin([t])]
        tt_order = df.sort_values(by=['lat', 'time'])['trainstation'].unique()

        # late minutes
        filename = options.save_path+'/detailed_heatmap/late_minutes_over_day_{}_{}.png'.format(name, dstr)
        s = df.groupby(['trainstation', df.index])['delay'].mean().to_frame()
        heatmap_series(s, locs, tt_order, filename)

        # total late ,minutes
        filename = options.save_path+'/detailed_heatmap/total_late_minutes_over_day_{}_{}.png'.format(name, dstr)
        s = df.groupby(['trainstation', df.index])['total_delay'].mean().to_frame()
        heatmap_series(s, locs, tt_order, filename)


def heatmap_year(l_data, passangers, year, locs):
    """
    Heatmap stations against day of the year

    l_data : DataFrame
            all labels data
    passangers : DataFrame
                passanger trains label data
    year : int
           year to handle
    """

    def heatmap_series(s, min_, max_, locs, order, filename):
        s = s.unstack(level=0)
        s.columns= s.columns.droplevel(0)
        s = s[order]
        s.set_index(s.index.dayofyear, inplace=True)
        s = s.groupby(s.index)[s.columns].mean()
        #df_norm = (s - s.mean()) / (s.max() - s.min())
        df_norm = 255*(s - min_) / (max_ - min_)
        #print(df_norm)
        viz.heatmap_train_station_delays(df_norm, locs, filename)


    logging.info('Processing year: {}'.format(year))
    #print(l_data)
    l_data = l_data[l_data.index.year==year]
    passangers = passangers[passangers.index.year==year]

    if len(l_data) == 0:
        return

    order = l_data.sort_values(by=['lat', 'time'])['trainstation'].unique()
    pass_order = passangers.sort_values(by=['lat', 'time'])['trainstation'].unique()

    # All trains, late minutes
    filename = options.save_path+'/heatmap/late_minutes_over_day_of_year_all_{}.png'.format(year)
    s = l_data.groupby(['trainstation', l_data.index])['delay'].mean().to_frame()
    max_ = s.max().values[0]
    min_ = s.min().values[0]
    heatmap_series(s, min_, max_, locs, order, filename)

    # All trains, total delay
    filename = options.save_path+'/heatmap/total_late_minutes_over_day_of_year_all_{}.png'.format(year)
    s = l_data.groupby(['trainstation', l_data.index])['total_delay'].mean().to_frame()
    heatmap_series(s, min_, max_, locs, order, filename)

    # Passanger trains, late minutes
    filename = options.save_path+'/heatmap/late_minutes_over_day_of_year_passanger_{}.png'.format(year)

    s = passangers.groupby(['trainstation', passangers.index])['delay'].mean().to_frame()
    heatmap_series(s, min_, max_, locs, pass_order, filename)

    # Passanger trains, total_delay
    filename = options.save_path+'/heatmap/total_late_minutes_over_day_of_year_passanger_{}.png'.format(year)

    s = passangers.groupby(['trainstation', passangers.index])['total_delay'].mean().to_frame()
    heatmap_series(s, min_, max_, locs, pass_order, filename)

    # for different train_types
    for name,t in train_types.items():

        try:
            df = l_data[l_data.loc[:,'train_type'].isin([t])]
            tt_order = df.sort_values(by=['lat', 'time'])['trainstation'].unique()

            # late minutes
            filename = options.save_path+'/heatmap/late_minutes_over_day_of_year_{}_{}.png'.format(name, year)
            s = df.groupby(['trainstation', df.index])['delay'].mean().to_frame()
            heatmap_series(s, min_, max_, locs, tt_order, filename)

            # total late ,minutes
            filename = options.save_path+'/heatmap/total_late_minutes_over_day_of_year_{}_{}.png'.format(name, year)
            s = df.groupby(['trainstation', df.index])['total_delay'].mean().to_frame()
            heatmap_series(s, min_, max_, locs, tt_order, filename)
        except pd.core.base.DataError as e:
            logging.error(e)


def main():
    """
    Get data from db and save it as csv
    """

    #a = mlfdb.mlfdb()
    a = _bq.BQHandler()
    io = _io.IO(gs_bucket='trains-data')
    viz = _viz.Viz()

    if not os.path.exists(options.save_path):
        os.makedirs(options.save_path)

    starttime, endtime = io.get_dates(options)

    logging.debug(options.what)
    what = options.what.split(',')
    logging.debug(what)

    all_param_names = ['time', 'trainstation', 'train_type', 'train_count', 'total_delay', 'delay', 'name', 'lat', 'lon']
    logging.info('Loading classification dataset from db')
    logging.info('Using time range {} - {}'.format(starttime.strftime('%Y-%m-%d'),
    endtime.strftime('%Y-%m-%d')))

    # Read data and filter desired train_types (ic and commuter)
    l_data = a.get_rows(starttime,
                       endtime,
                       loc_col='trainstation',
                       project='trains-197305',
                       dataset='trains_2009_18',
                       table='features',
                       parameters=all_param_names)

    # data = io.filter_train_type(labels_df=data,
    #                             train_types=['K','L'],
    #                             sum_types=True,
    #                             train_type_column='train_type',
    #                             location_column='trainstation',
    #                             time_column='time',
    #                             sum_columns=['delay'],
    #                             aggs=aggs)

    # l_data.rename(columns={0: 'trainstation', 1:'time', 2: 'lon', 3: 'lat', 4: 'train type', 5: 'delay', 6: 'train count', 7: 'total delay'}, inplace=True)

    #l_data.set_index(pd.DatetimeIndex(pd.to_datetime(l_data.loc[:,'time'].astype(int), unit='s')), inplace=True)
    #l_data.set_index('time', drop=False, inplace=True)

    passangers = io.filter_train_type(labels_df=l_data, train_types=['L','K'], sum_types=True)
    l_data.set_index(pd.to_datetime(l_data.loc[:,'time']), inplace=True)
    #passangers.set_index(pd.to_datetime(passangers.loc[:,'time']), inplace=True)


    # ################################################################################
    if 'histograms' in what:

        # All delays
        filename = options.save_path+'/hist_all_delays_all.png'
        viz.hist_all_delays(l_data.loc[:,['train_type', 'train_count', 'delay', 'total_delay']], filename)

        # Different train types

        for name,t in train_types.items():
            filename = options.save_path+'/hist_all_delays_{}.png'.format(name)
            df = l_data[l_data.loc[:,'train_type'].isin([t])]
            viz.hist_all_delays(df.loc[:,statlist], filename)

        # All passanger trains
        filename = options.save_path+'/hist_all_delays_passanger.png'
        viz.hist_all_delays(passangers.loc[:,statlist], filename)

        # all parameters
        passangers.replace(-99, np.nan, inplace=True)
        delayed_data = passangers[passangers.loc[:,'delay'] > 50]
        d = {'A': passangers, 'B': delayed_data}
        comp_data = pd.concat(d.values(), axis=1, keys=d.keys())
        filename = options.save_path+'/histograms_compare.png'
        viz.all_hist(comp_data, filename=filename)

    # ################################################################################
    if 'history' in what:

        # Mean delays over time

        # All trains
        filename = options.save_path+'/mean_delays_over_time_all.png'
        s = l_data.groupby(l_data.index)[statlist].mean()
        viz.plot_delays(s, filename)

        # for passanger trains
        filename = options.save_path+'/mean_delays_over_time_passanger.png'
        s = passangers.groupby(passangers.index)[statlist].mean()
        viz.plot_delays(s, filename)

        # for different train_types
        for name,t in train_types.items():
            filename = options.save_path+'/mean_delays_over_time_{}.png'.format(name)
            df = l_data[l_data.loc[:,'train_type'].isin([t])]
            s = df.groupby(df.index)[statlist].mean()
            viz.plot_delays(s, filename)


        # Median delays over time

        # All trains
        filename = options.save_path+'/median_delays_over_time_all.png'
        s = l_data.groupby(l_data.index)[statlist].median()
        viz.plot_delays(s, filename)

        # for passanger trains
        filename = options.save_path+'/median_delays_over_time_passanger.png'
        s = passangers.groupby(passangers.index)[statlist].median()
        viz.plot_delays(s, filename)

        # for different train_types
        for name,t in train_types.items():
            filename = options.save_path+'/median_delays_over_time_{}.png'.format(name)
            df = l_data[l_data.loc[:,'train_type'].isin([t])]
            s = df.groupby(df.index)[statlist].median()
            viz.plot_delays(s, filename)

    # ################################################################################
    if 'heatmap' in what:

        # locs = a.get_locations_by_dataset(options.dataset,
        #                                   starttime=starttime,
        #                                   endtime=endtime,
        #                                   rettype='dict')
        # # Heatmap bad some stations
        #locs = l_data.loc[:, 'trainstation'].unique().values.ravel()
        locs = io.get_train_stations('cnf/stations.json')
        #print(locs)

        if not os.path.exists(options.save_path+'/heatmap'):
            os.makedirs(options.save_path+'/heatmap')

        heatmap_year(l_data, passangers, 2018, locs)

        for year in np.arange(2010, 2019, 1):
            heatmap_year(l_data, passangers, year, locs)

    # ################################################################################
    if 'detailed_heatmap' in what:
        locs = a.get_locations_by_dataset(options.dataset,
                                          starttime=starttime,
                                          endtime=endtime,
                                          rettype='dict')
        # Heatmap bad some stations


        if not os.path.exists(options.save_path+'/detailed_heatmap'):
            os.makedirs(options.save_path+'/detailed_heatmap')

        d = starttime
        while d < endtime:
            heatmap_day(l_data, passangers, d, locs)
            d += dt.timedelta(days=1)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, help='End time of the classification data interval')
    parser.add_argument('--save_path', type=str, default=None, help='Save path')
    parser.add_argument('--what', type=str, default='histograms,history', help='What to plot')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
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
