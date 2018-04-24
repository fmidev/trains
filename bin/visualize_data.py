import sys
import os
import argparse
import logging
import datetime as dt
import json
import itertools
import numpy as np
import pandas as pd

from mlfdb import mlfdb
from ml_feature_db.api.mlfdb import mlfdb as db
from lib import io as _io
from lib import viz as _viz

from multiprocessing import Pool, cpu_count

statlist = ['late minutes', 'total late minutes']
train_types = {'Intercity': 0,
               'Commuter': 1,
               'Cargo': 2,
               'Other': 3}

a = db.mlfdb()
io = _io.IO()
viz = _viz.Viz()


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

    order = l_data.sort_values(by=['lat', 'time'])['location id'].unique()
    pass_order = passangers.sort_values(by=['lat', 'time'])['location id'].unique()
    
    if len(l_data) == 0:
        return
            
    # All trains, late minutes
    filename = options.save_path+'/detailed_heatmap/late_minutes_over_day_all_{}.png'.format(dstr)

    df = l_data.groupby(['location id', l_data.index])['late minutes'].mean().to_frame()
    heatmap_series(df, locs, order, filename)

    # All trains, total late minutes
    filename = options.save_path+'/detailed_heatmap/total_late_minutes_over_day_all_{}.png'.format(dstr)
    s = l_data.groupby(['location id', l_data.index])['total late minutes'].mean().to_frame()
    heatmap_series(s, locs, order, filename)
        
    # Passanger trains, late minutes
    filename = options.save_path+'/detailed_heatmap/late_minutes_over_day_passanger_{}.png'.format(dstr)
        
    s = passangers.groupby(['location id', passangers.index])['late minutes'].mean().to_frame()
    heatmap_series(s, locs, pass_order, filename)
        
    # Passanger trains, total late minutes
    filename = options.save_path+'/detailed_heatmap/total_late_minutes_over_day_passanger_{}.png'.format(dstr)
    
    s = passangers.groupby(['location id', passangers.index])['total late minutes'].mean().to_frame()
    heatmap_series(s, locs, pass_order, filename)

    # for different train types
    for name,t in train_types.items():

        df = l_data[l_data.loc[:,'train type'].isin([t])]
        tt_order = df.sort_values(by=['lat', 'time'])['location id'].unique()
        
        # late minutes
        filename = options.save_path+'/detailed_heatmap/late_minutes_over_day_{}_{}.png'.format(name, dstr)
        s = df.groupby(['location id', df.index])['late minutes'].mean().to_frame()
        heatmap_series(s, locs, tt_order, filename)

        # total late ,minutes
        filename = options.save_path+'/detailed_heatmap/total_late_minutes_over_day_{}_{}.png'.format(name, dstr)
        s = df.groupby(['location id', df.index])['total late minutes'].mean().to_frame()
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

    def heatmap_series(s, locs, order, filename):
        s = s.unstack(level=0)
        s.columns= s.columns.droplevel(0)
        s = s[order]
        s.set_index(s.index.dayofyear, inplace=True)
        s = s.groupby(s.index)[s.columns].mean()
        viz.heatmap_train_station_delays(s, locs, filename)

        
    logging.info('Processing year: {}'.format(year))
    l_data = l_data[str(year)]
    passangers = passangers[str(year)]
    
    if len(l_data) == 0:
        return

    order = l_data.sort_values(by=['lat', 'time'])['location id'].unique()
    pass_order = passangers.sort_values(by=['lat', 'time'])['location id'].unique() 
       
    # All trains, late minutes
    filename = options.save_path+'/heatmap/late_minutes_over_day_of_year_all_{}.png'.format(year)
    s = l_data.groupby(['location id', l_data.index])['late minutes'].mean().to_frame()
    heatmap_series(s, locs, order, filename)
    
    # All trains, total late minutes
    filename = options.save_path+'/heatmap/total_late_minutes_over_day_of_year_all_{}.png'.format(year)
    s = l_data.groupby(['location id', l_data.index])['total late minutes'].mean().to_frame()
    heatmap_series(s, locs, order, filename)
        
    # Passanger trains, late minutes
    filename = options.save_path+'/heatmap/late_minutes_over_day_of_year_passanger_{}.png'.format(year)
        
    s = passangers.groupby(['location id', passangers.index])['late minutes'].mean().to_frame()
    heatmap_series(s, locs, pass_order, filename)
        
    # Passanger trains, total late minutes
    filename = options.save_path+'/heatmap/total_late_minutes_over_day_of_year_passanger_{}.png'.format(year)
    
    s = passangers.groupby(['location id', passangers.index])['total late minutes'].mean().to_frame()
    heatmap_series(s, locs, pass_order, filename)

    # for different train types
    for name,t in train_types.items():

        df = l_data[l_data.loc[:,'train type'].isin([t])]
        tt_order = df.sort_values(by=['lat', 'time'])['location id'].unique()
        
        # late minutes
        filename = options.save_path+'/heatmap/late_minutes_over_day_of_year_{}_{}.png'.format(name, year)
        s = df.groupby(['location id', df.index])['late minutes'].mean().to_frame()
        heatmap_series(s, locs, tt_order, filename)

        # total late ,minutes
        filename = options.save_path+'/heatmap/total_late_minutes_over_day_of_year_{}_{}.png'.format(name, year)
        s = df.groupby(['location id', df.index])['total late minutes'].mean().to_frame()
        heatmap_series(s, locs, tt_order, filename)


def main():
    """
    Get data from db and save it as csv
    """

    #a = mlfdb.mlfdb()
    
    if not os.path.exists(options.save_path):
        os.makedirs(options.save_path)

    starttime, endtime = io.get_dates(options)

    logging.debug(options.what)
    what = options.what.split(',')
    logging.debug(what)
        
    logging.info('Loading classification dataset from db')
    if starttime is not None and endtime is not None:
        logging.info('Using time range {} - {}'.format(starttime.strftime('%Y-%m-%d'), endtime.strftime('%Y-%m-%d')))        


    l_data = a.get_rows(options.dataset,
                        starttime=starttime,
                        endtime=endtime,
                        rowtype='label',
                        return_type='pandas',
                        parameters=[])

    logging.debug('Data loaded')

    l_data.loc[:,4] = l_data.loc[:,4].astype(int)
    l_data.loc[:,5] = l_data.loc[:,5].astype(int)
    l_data.loc[:,6] = l_data.loc[:,6].astype(int)
    l_data.loc[:,7] = l_data.loc[:,7].astype(int)
    
    l_data.rename(columns={0: 'location id', 1:'time', 2: 'lon', 3: 'lat', 4: 'train type', 5: 'late minutes', 6: 'train count', 7: 'total late minutes'}, inplace=True)

    l_data.set_index(pd.DatetimeIndex(pd.to_datetime(l_data.loc[:,'time'].astype(int), unit='s')), inplace=True)
    
    passangers = io.filter_train_type(labels_df=l_data, traintypes=[0,1], sum_types=True)
    passangers.set_index(pd.DatetimeIndex(pd.to_datetime(passangers.loc[:,'time'].astype(int), unit='s')), inplace=True)


    # ################################################################################
    if 'histograms' in what:

        # All delays
        filename = options.save_path+'/hist_all_delays_all.png'
        viz.hist_all_delays(l_data.loc[:,['train type', 'train count', 'late minutes', 'total late minutes']], filename)
        
        # Different train types

        for name,t in train_types.items():
            filename = options.save_path+'/hist_all_delays_{}.png'.format(name)
            df = l_data[l_data.loc[:,'train type'].isin([t])]
            viz.hist_all_delays(df.loc[:,statlist], filename)
    
        # All passanger trains
        filename = options.save_path+'/hist_all_delays_passanger.png'
        viz.hist_all_delays(passangers.loc[:,statlist], filename)
        
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

        # for different train types
        for name,t in train_types.items():
            filename = options.save_path+'/mean_delays_over_time_{}.png'.format(name)
            df = l_data[l_data.loc[:,'train type'].isin([t])]
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

        # for different train types
        for name,t in train_types.items():
            filename = options.save_path+'/median_delays_over_time_{}.png'.format(name)
            df = l_data[l_data.loc[:,'train type'].isin([t])]
            s = df.groupby(df.index)[statlist].median()
            viz.plot_delays(s, filename)

    # ################################################################################
    if 'heatmap' in what:
        
        locs = a.get_locations_by_dataset(options.dataset,
                                          starttime=starttime,
                                          endtime=endtime,
                                          rettype='dict')
        # Heatmap bad some stations 


        if not os.path.exists(options.save_path+'/heatmap'):
            os.makedirs(options.save_path+'/heatmap')

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
