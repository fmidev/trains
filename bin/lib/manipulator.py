# -*- coding: utf-8 -*-
import sys, os, json, logging, joblib
#from sklearn.externals import joblib
import numpy as np
import pandas as pd


class Manipulator:

    def __init__(self):
        pass

    def pred_fractiles(self, metadata, pred, stationList):
        """
        Create fractiles from prediction

        metadata    : Pandas dataframe
                      pandas dataframe with 'trainstation' and 'time columns'
        pred        : lst
                      predicted delay in the same order with metadata
        stationList : struct
                      {stationName : {'lat': xx, 'lon': xx}}

        return      : Pandas Dataframe
                      'trainstation', 'time', 'delay', 'lower bound', 'upper bound'
        """

        def group(row):
            """
            Divide stations to three groups based on their latitude (latitudes
            got from https://ilmatieteenlaitos.fi/saaennusteiden-aluejako)
            """
            if stationList[row]['lat'] > 63.41: return 3
            if stationList[row]['lat'] > 61.55: return 2
            if stationList[row]['lat'] > 59: return 1
            return 0

        # Divide stations to groups
        df = pd.concat([metadata.reset_index(), pd.DataFrame(pred, columns=['pred_delay'])], axis=1)
        df['group'] = df.apply(lambda x: group(x['trainstation']), axis=1)
        df.sort_values(by=['time', 'trainstation'], inplace=True)

        indexed = df.set_index(keys=['time'])

        avg = pd.DataFrame(columns=['avg_delay', 'avg_pred_delay', 'avg_pred_delay_low', 'avg_pred_delay_high'])
        avg.loc[:,'avg_delay'] = indexed.loc[:,'delay'].groupby('time').mean()
        avg.loc[:,'avg_pred_delay'] = indexed.loc[:,'pred_delay'].groupby('time').mean()

        groups = []
        for i in np.arange(0,3):
            gdf = pd.DataFrame(columns=['avg_delay', 'avg_pred_delay', 'avg_pred_delay_low', 'avg_pred_delay_high'])
            gdf.loc[:,'avg_delay'] = indexed.loc[indexed['group']==(i+1),'delay'].groupby('time').mean()
            gdf.loc[:,'avg_pred_delay'] = indexed.loc[indexed['group']==(i+1),'pred_delay'].groupby('time').mean()
            gdf.loc[:,'avg_pred_delay_low'] = indexed.loc[indexed['group']==(i+1),'pred_delay'].groupby('time').quantile(.1)
            gdf.loc[:,'avg_pred_delay_high'] = indexed.loc[indexed['group']==(i+1),'pred_delay'].groupby('time').quantile(.9)

            s = indexed.loc[indexed['group']==(i+1),'pred_delay'].groupby('time').quantile(.05).to_frame()
            s2 = indexed.loc[indexed['group']==(i+1),'pred_delay'].groupby('time').quantile(.95).to_frame()
            for t in s.index:
                time_mask = (df['time'] == t)
                mask = time_mask & (df['group'] == (i+1))
                df.loc[mask, 'pred_delay_low'] = s.loc[t, 'pred_delay']
                df.loc[mask, 'pred_delay_high'] = s2.loc[t, 'pred_delay']

            groups.append(gdf.copy(True))

        return groups, avg, df

    def read_parameters(self, filename, drop=0):
        """
        Read parameter from file (one param per line)

        filename : str
                   filename
        """

        file_to_open = filename

        with open(file_to_open, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        params, names = [], []
        for line in lines:
            param, name = line.split(';')
            params.append(param)
            names.append(name)

        if drop > 0:
            params = params[drop:]
            names = names[drop:]

        return params, names


    def get_train_stations(self, filename=None, key='short'):

        """
        Get railway stations from file or db digitrafic
        """
        if filename is None:
            url = "https://rata.digitraffic.fi/api/v1/metadata/stations"

            with urllib.request.urlopen(url) as u:
                data = json.loads(u.read().decode("utf-8"))
        else:
            with open(filename) as f:
                data = json.load(f)

        stations = dict()

        if key == 'short':
            for s in data:
                latlon = {'lat': s['latitude'], 'lon': s['longitude'], 'name': s['stationName']}
                stations[s['stationShortCode'].encode('utf-8').decode()] = latlon
        elif key == 'long':
            for s in data:
                latlon = {'lat': s['latitude'], 'lon': s['longitude'], 'name': s['stationShortCode']}
                stations[s['stationName'].encode('utf-8').decode()] = latlon

        return stations


    def _calc_prec_sums(self, obs, prec_column=5):
        """
        Calculate 3h and 6h prec sums (private method)
        """
        sum_3h = []
        sum_6h = []

        obs.loc[:,'3hsum'] = -99
        obs.loc[:,'6hsum'] = -99

        for h,values in obs.iterrows():
            prec = values[prec_column]
            if prec < 0: prec = 0
            sum_3h.append(prec)
            sum_6h.append(prec)

            if len(sum_3h) > 3:
                sum_3h.pop(0)
                _sum = sum(sum_3h)
                if _sum < 0: _sum = -99
                obs.loc[h, '3hsum'] = _sum

            if len(sum_6h) > 6:
                sum_6h.pop(0)
                _sum = sum(sum_6h)
                if _sum < 0: _sum = -99
                obs.loc[h, '6hsum'] = _sum

        return obs


    def load_scikit_model(self, filename):
        """
        Load scikit model from file
        """
        logging.info('Loading model from file {}...'.format(filename))
        return joblib.load(str(filename))

    def get_batch_size(self, data, pad_strategy='pad', quantile=None):
        """
        Get batch size based on pad_strategy. See extract_batch docs for more info.
        """
        if pad_strategy == 'sample':
            return min(data.groupby(['time']).size())
        elif pad_strategy == 'pad':
            return max(data.groupby(['time']).size())
        elif pad_strategy == 'drop':
            #logging.debug(data.groupby(['time']).size())
            return int(data.groupby(['time']).size().quantile(quantile))

    def pad_along_axis(self, a, target_length, constant_values=-99, axis=0):
        """
        Pad along given axis
        """
        pad_size = target_length - a.shape[axis]
        axis_nb = len(a.shape)

        if pad_size < 0:
            return a

        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)

        b = np.pad(a, pad_width=npad, mode='constant', constant_values=constant_values)

        return b

    def extract_batch(self, data, n_timesteps, batch_num=0, batch_size=None, pad_strategy='pad', quantile=None, start=None, end=None, feature_params=[], label_params=[]):
        """
        Extract and/or preprocess batch from data.

        data : pd.DataFrame
               Data
        n_timesteps : int
                      Number of timesteps to be used in LSTM. If <0 all timesteps are used-
        batch_num : int
                    Batch number, default 0
        batch_size : int or None
                     If None, batch size is got based on pad_strategy.
                     If <0, all valid values are returned (used specially for test data)
                     If >0, given batch size is used.
        pad_strategy : str
                       - If 'pad', largest timestep (with most stations) is chosen
                       from the data and other timesteps are padded with -99
                       - If 'sample', smallest timestep is chosen and other timesteps
                       are sampled to match smallest one.
                       - If 'drop', batch_size is chosen to to match options.quantile
                       value. If for example options.quantile=0.3 70 percent of timesteps
                       are taken into account.
        quantile : float or None
                   Used to drop given fraction of smaller timesteps from data if
                   pad_strategy='drop'

        return : (Values shape: (n_timesteps, batch_size, n_features), Labels shape (n_timesteps, batch_size))
        """
        # Detect batch size
        if batch_size is None:
            batch_size = self.get_batch_size(data, pad_strategy, quantile)
        elif batch_size < 0:
            #batch_size = len(data)
            batch_size = len(data.trainstation.unique())

        # print(batch_size)
        # If pad_strategy is drop, drop timesteps with too few stations
        if pad_strategy == 'drop':
            t_size = data.groupby(['time']).size()
            t_size = t_size[t_size >= batch_size].index.values
            data = data[data.time.isin(t_size)]

        all_times = data.time.unique()
        stations = data.trainstation.unique()

        if n_timesteps < 0:
            n_timesteps = len(all_times)

        # Pick times for the batch
        if start is None:
            start = batch_num*n_timesteps
        if end is None:
            end = start + n_timesteps

        times = all_times[start:end]

        values = []
        labels = []

        # Go through times
        for t in times:
            # Pick data for current time
            timestep_values = data[data.loc[:,'time'] == t]

            # If pad_strategy is drop and timestep is too small, ignore it and continue
            if pad_strategy == 'drop' and batch_size > len(timestep_values):
                continue

            # If pad_strategy is sample or drop, sample data to match desired batch_size
            if pad_strategy in ['sample','drop'] and batch_size < len(timestep_values):
                timestep_values = timestep_values.sample(batch_size)

            # pd dataframe to np array
            #timestamp_values = timestep_values.drop(columns=['time', 'trainstation', 'delay', 'train_type']).astype(np.float32).values
            timestamp_values = timestep_values.loc[:, feature_params].astype(np.float32).values
            label_values = timestep_values.loc[:, label_params].astype(np.float32).values.ravel()

            #print(pd.DataFrame(label_values))
            # if pad_strategy is pad and timestep is too small, pad it with -99
            if pad_strategy == 'pad' and batch_size > len(timestamp_values):
                timestamp_values = self.pad_along_axis(timestamp_values, batch_size, -99)
                label_values = self.pad_along_axis(label_values, batch_size, -99)

            values.append(timestamp_values)
            labels.append(label_values)

        values = np.array(values)
        labels = np.array(labels)

        # print(batch_size)
        # print(values.shape)
        # print(labels.shape)
        # print(labels)
        # sys.exit()

        #values = np.rollaxis(np.array(values), 1)
        #labels = np.reshape(np.rollaxis(np.array(labels), 1), (batch_size, n_timesteps, 1))

        logging.debug('Values shape: {}'.format(values.shape))
        logging.debug('Labels shape: {}'.format(labels.shape))

        return values, labels
