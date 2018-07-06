# -*- coding: utf-8 -*-
import sys, re, tempfile, subprocess
import numpy as np
from os import listdir
from os.path import isfile, join

from google.cloud import storage

from sklearn.externals import joblib
from keras.models import Model, model_from_yaml
import tensorflow as tf

import googleapiclient.discovery

import boto3
import random
import pickle
import json,csv
import logging
import datetime as dt
import pandas as pd

class IO:

    s3 = False
    gs = False
    bucket_name = ''
    bucket = ''
    client = ''

    def __init__(self, s3_bucket=None, gs_bucket=None):

        if s3_bucket is not None:
            self.bucket_name = s3_bucket
            self.s3 = True
            self.client = boto3.client('s3')
            resource = boto3.resource('s3')
            self.bucket = resource.Bucket(self.bucket_name)
        elif gs_bucket is not None:
            self.bucket_name = gs_bucket
            self.gs = True

    def _upload_to_bucket(self, filename, ext_filename):
        if self.s3:
            raise ValueError('S3 not implemented')
        if self.gs:
            client = storage.Client()
            bucket = client.get_bucket(self.bucket_name)
            blob = storage.Blob(ext_filename, bucket)
            blob.upload_from_filename(filename)
            logging.info('Uploaded to {}'.format(ext_filename))

    #
    # GENERAL
    #
    def write_csv(self, _dict, filename, ext_filename=None):
        """
        Write dict to csv

        _dict : dict
                data in format ['key': [values], 'key2': values]
        filename : str
                   filename where data is saved
        """
        with open(filename, 'w') as f:
            f.write('"'+'";"'.join(_dict.keys())+'"\n')
            for i in np.arange(len(_dict[list(_dict.keys())[0]])):
                values = []
                for col in _dict.keys():
                    values.append(str(_dict[col][i]))
                f.write(';'.join(values)+'\n')
        logging.info('Wrote {}'.format(filename))
        self._upload_to_bucket(filename, ext_filename)

    def dict_to_csv(self, dict, filename):
        """
        Save given dict to csv file
        """
        with open(filename, 'w') as f:
            w = csv.DictWriter(f, dict.keys())
            w.writeheader()
            w.writerow(dict)
        logging.info('Wrote {}'.format(filename))
        self._upload_to_bucket(filename, ext_filename)


    def dict_to_json(self, dict, filename):
        """
        Save given dict to json file
        """
        with open(filename, 'w') as f:
            f.write(json.dumps(dict))
        logging.info('Wrote {}'.format(filename))
        self._upload_to_bucket(filename, ext_filename)


    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def read_data(self, xfilename, yfilename=None,
                  data_type=None, delimiter=';',
                  skip_cols=0, skip_rows=1,
                  force_local=False, remove=None):
        """
        Read data from csv files
        """
        X, y = [], []

        # Read data
        if self.s3 and not force_local:
            tmp = tempfile.NamedTemporaryFile()
            self.bucket.download_file(xfilename, tmp.name)
            with open(tmp.name) as f:
                lines = f.read().splitlines()
        else:
            with open(xfilename, encoding='utf-8') as f:
                lines = f.read().splitlines()

        for line in lines[skip_rows:]:
            l = line.split(delimiter)[skip_cols:]
            if remove is not None:
                nl = []
                for e in l:
                    nl.append(e.replace(remove, ''))
                l = nl
                if data_type is not None:
                    l = list(map(data_type, l))

            X.append(l)

        try:
            if self.s3 and not force_local:
                tmp = tempfile.NamedTemporaryFile()
                self.bucket.download_file(yfilename, tmp.name)
                with open(tmp.name) as f:
                    lines = f.read().splitlines()
            else:
                with open(yfilename) as f:
                    lines = f.read().splitlines()

            for line in lines[skip_rows:]:
                y.append(list(map(data_type, line)))
        except:
            return X

        return X, y

    def pick_set(self, x, y, num_of_samples):
        """ Split data to train and validation data set
        x:               [list, n,k]  feature list (samples as rows, features as cols)
        y:               [list, n,1]  label list
        num_of_samples:  [int]        number of samplest to pick
        """

        selected, ids, set_x, set_y = [], [], [], []
        r = random.SystemRandom()

        if len(set_x) >= num_of_samples:
            num_of_samples = len(set_x)-1

        while len(set_x) < num_of_samples:
            i = r.randrange(0, len(x))
            set_x.append(x.pop(i))
            set_y.append(y.pop(i))
            ids.append(i)

        set_x = np.matrix(set_x)
        set_y = np.matrix(set_y).T

        return ids, set_x, set_y

    def shift_v(self, v, shift=-1):
        new_v = []
        for i in v:
            new_v.append(i+shift)
        return new_v


    def read_parameters(self, filename, force_local=False, drop=0):
        """
        Read parameter from file (one param per line)

        filename : str
                   filename
        """

        file_to_open = filename
        if self.s3 and not force_local:
            tmp_m = tempfile.NamedTemporaryFile()
            self.bucket.download_file(filename, tmp_m.name)
            file_to_open=tmp_m.name

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

    def save_scikit_model(self, model, filename, ext_filename=None):
        """
        Save scikit model into file
        """
        joblib.dump(model, filename)
        logging.info('Saved model into {}'.format(filename))
        self._upload_to_bucket(filename, ext_filename)

    def load_scikit_model(self, filename):
        """
        Load scikit model from file
        """

        if self.s3 and not force_local:
            raise ValueError('S3 not implemented')
        if self.gs and not force_local:
            client = storage.Client()
            bucket = client.get_bucket(self.bucket_name)
            tmp = tempfile.NamedTemporaryFile()
            blob = storage.Blob(filename, bucket)
            blob.download_to_filename(str(tmp))
        else:
            tmp = filename

        return joblib.load(str(tmp))

    def save_model(self, model_filename, weights_filename, history_filename, model, history):
        """
        Save keras model and weights into file
        """
        print("Saving model into {} and weights into {}".format(model_filename, weights_filename))
        with open(model_filename, 'w') as f:
            f.write(model.to_yaml())
            model.save_weights(weights_filename)

        with open(history_filename, 'wb') as f:
            pickle.dump(history, f)

        if self.s3:
            self.bucket.upload_file(model_filename, model_filename)
            self.bucket.upload_file(weights_filename, weights_filename)
            self.bucket.upload_file(history_filename, history_filename)


    def load_model(self, model_filename, weights_filename, history_filename, force_local=False):
        """
        Load keras model and weights from file
        """

        model_file_to_open = model_filename
        weights_file_to_open = weights_filename
        history_file_to_open = history_filename
        if self.s3 and not force_local:
            tmp_m = tempfile.NamedTemporaryFile()
            self.bucket.download_file(model_filename, tmp_m.name)
            model_file_to_open=tmp_m.name

            tmp_w = tempfile.NamedTemporaryFile()
            self.bucket.download_file(weights_filename, tmp_w.name)
            weights_file_to_open = tmp_w.name

            tmp_h = tempfile.NamedTemporaryFile()
            self.bucket.download_file(history_filename, tmp_h.name)
            history_file_to_open = tmp_h.name

        with open(model_file_to_open, 'r') as f:
            model = model_from_yaml(f.read())
        model.load_weights(weights_file_to_open)
        # self.pred_fun = get_layer_output_function(model, 'output_realtime')
        print('Model loaded')

        with open(history_file_to_open, 'rb') as f:
            history = pickle.load(f)

        return model, history

    def export_tf_model(self, sess, export_dir, inputs, outputs, serving_input_fn=None):
        """
        Save tensor flow model
        """
        logging.info('Exporting tf model to {}...'.format(export_dir))
        tf.estimator.Estimator.export_savedmodel(
            export_dir_base=export_dir,
            serving_input_receiver_fn=serving_input_fn
        )


    def get_files_to_process(self, path, suffix='csv', force_local=False):
        """
        List files in given directory location. If google cloud bucket is
        given, use that. Else, use filesystem. NOTE S3 not implemented.

        path : string
               path to find for files
        suffix : string
                 file suffix

        return list of files
        """

        if self.s3 and not force_local:
            raise ValueError('S3 not implemented')
        if self.gs and not force_local:
            client = storage.Client()
            bucket = client.get_bucket(self.bucket_name)

            files = [o.name for o in bucket.list_blobs()
                     if o.name.startswith(path) and o.name.endswith(suffix)]

        else:
            files = []
            for file in os.listdir(path):
                if file.endswith(suffix):
                    files.append(os.path.join(dir, file))

        return files

    def get_file_to_process(self, filename, force_local=False):
        """
        Download file from bucket for processing. If bucket is not
        set, local file is returned

        filename : String
                   filename with full path
        force_local : boolean
                      ignore buckets (default False)

        return tmp filename
        """

        if self.s3 and not force_local:
            raise ValueError('S3 not implemented')
        if self.gs and not force_local:
            client = storage.Client()
            bucket = client.get_bucket(self.bucket_name)
            tmp = tempfile.NamedTemporaryFile()
            blob = storage.Blob(filename, bucket)
            blob.download_to_filename(str(tmp))
        else:
            return filename

        return str(tmp)

    def get_dates(self, options):
        """
        Parse DateTime objects from options

        options : dict
                  options from argparse

        return starttime (DateTime), endtime (DateTime)
        """
        try:
            starttime = dt.datetime.strptime(options.starttime, "%Y-%m-%d")
        except:
            starttime = None
            logging.info('Starttime not set or malformed')

        try:
            endtime = dt.datetime.strptime(options.endtime, "%Y-%m-%d")
        except:
            endtime = None
            logging.info('Endtime not set or malformed')

        return starttime, endtime


    #
    # TRAINS
    #
    def station_names_to_abb(self,
                             df,
                             locations,
                             location_column='loc_name',
                             stations_file='cnf/stations.json'):
        """
        Change long names to short ones.
        """
        stations = self.get_train_stations(stations_file, key='long')

        def change(loc_name):
            return stations[loc_name]['name']

        df[location_column] = df.apply(lambda row: change(row[location_column]), axis=1)
        return df


    def station_ids_to_names(self, df, locations, location_column='loc_id',
                             stations_file=None):
        """
        Change location id to station name in the given DataFrame

        df                : DataFrame
                            DataFrame to handle
        location_column   : str
                            name of the location column
        stations_filename : str
                            file where stations are stored, if None digitrafic is used

        returns DataFrame
        """

        #stations = self.get_train_stations(stations_file)
        locs_cache = {}
        #print(stations)

        def change(loc_id):
            loc_name='unkown'
            if loc_id in locs_cache:
                loc_name = locs_cache[loc_id]
            else:
                for l in locations:
                    if loc_id == l[0]:
                        loc_name=l[1]
                        locs_cache[loc_id] = loc_name
                        break

            return loc_name

        df[location_column] = df.apply(lambda row: change(row[location_column]), axis=1)
        return df

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

    def find_id(self, locations, name):
        """
        Find id from (id, name) tuple list
        """
        for loc in locations:
            if str(name) == str(loc[1]):
                return loc[0]

        raise KeyError('Id for name {} not found'.format(name))

    def filter_labels(self, l_data, f_data,
                      invert=False, uniq=False,
                      location_column='loc_name', time_column='time'):
        """
        Return labels which have corresponding features

        labels_metadata : list
                          labels metadata
        labels : numpy array
                 labels
        features_metadata : list
                            features metadata
        features : numpy array
                   features
        invert : bool
                 if True, labels with missing features are returned (default False)

        return filtered labels_metadata, labels (numpy array)
        """

        logging.debug('Filtering labels...')
        if len(f_data) == 0 or len(l_data) == 0:
            return l_data

        l_data.columns = l_data.columns.map(lambda x: str(x) + '_label')
        # print(l_data.loc[l_data['loc_name_label'] == 'KOK'])
        # print(f_data.loc[f_data['loc_name'] == 'KOK'])
        #sys.exit()
        f_l_data = pd.merge(l_data,
                            f_data, #.drop(columns=[2,3]),
                            left_on=[location_column+'_label',time_column+'_label'],
                            right_on=[location_column, time_column],
                            how='left',
                            #left_index=True,
                            indicator=True,
                            copy=False)
        #print(f_l_data[['late_minutes_label', 'train_count_label', 'total_late_minutes_label', 'lat_label', 'lon_label', 'loc_name_label', 'time_label']])

        if uniq:
            f_l_data.drop_duplicates(subset=[location_column+'_label', time_column+'_label'], inplace=True)

        if invert:
            f_l_data = f_l_data.loc[(f_l_data['_merge'] == 'left_only')]
        else:
            f_l_data = f_l_data.loc[(f_l_data['_merge'] == 'both')]

        filter_cols = []
        rename={}
        for col in f_l_data:
            try:
                if col.endswith('_label'):
                    filter_cols.append(col)
                    rename[col] = col[:-6]
            except AttributeError as e:
                pass

        l_data = f_l_data[filter_cols]
        l_data.rename(columns=rename, inplace=True)

        f_data = f_l_data.drop(columns=filter_cols+['_merge'])

        logging.debug('Shape of feature data: {}'.format(f_data.shape))
        logging.debug('Shape of filtered data: {}'.format(l_data.shape))

        assert len(l_data) == len(f_data)

        return l_data, f_data

    def uniq(self, metadata, data):
        """
        Drop uniq values from metadata and data

        metadata : np.array like
                   metadata
        data : np.array
               data

        returns : metadata, data
        """

        df = pd.DataFrame(metadata)
        mask = np.invert(df.duplicated([0,1]).as_matrix())
        data_metadata = metadata[(mask)]
        data = data[(mask)]
        logging.debug('Shape of uniq metadata: {}'.format(metadata.shape))
        logging.debug('Shape of uniq data: {}'.format(data.shape))
        return metadata, data

    def get_aggs_from_params(self, params):
        """
        Get aggregation type from SmartMet Server param param names

        params : list
                 list of params

        return dict which can be give for pandas agg function
        """
        aggs = {}
        possible_aggs = ['min', 'max', 'mean', 'sum', 'median']
        for param in params:
            agg = param.split('(')[0]
            if agg in possible_aggs:
                aggs[param] = agg
            else:
                aggs[param] = 'mean'
        return aggs

    def filter_train_type(self, labels_df=[],
                          train_types=[],
                          sum_types = False,
                          train_type_column='train_type',
                          location_column='trainstation',
                          time_column='time',
                          sum_columns=['train count', 'late minutes', 'total late minutes'],
                          aggs=[]):
        """
        Filter traintypes from metadata

        train_types     : list
                          list of following options: [0,1,2,3]

                          train_types = {'K': 0,
                                         'L': 1,
                                         'T': 2,
                                         'M': 3}
        sum_types       : bool
                         if True, sum different train types together (default False)

        returns : np array, np array
                  labels metadata, labels
        """

        if len(labels_df) == 0:
            return labels_df

        mask = labels_df.loc[:,train_type_column].isin(train_types)

        filt_labels_df = labels_df[(mask)]

        if sum_types:
            d = {}
            for col in sum_columns:
                d[col] = ['sum']
            for col,method in aggs.items():
                d[col] = [method]
            d['lat'] = ['max']
            d['lon'] = ['max']

            filt_labels_df = filt_labels_df.groupby([location_column, time_column], as_index=False).agg(d)
            #print(filt_labels_df)

            filt_labels_df.columns = filt_labels_df.columns.droplevel(1)
            filt_labels_df.drop_duplicates([location_column, time_column], inplace=True)
            #print(filt_labels_df)

        return filt_labels_df


    def filter_labels_by_latlon(self, labels_metadata, latlon):
        """
        Filter labels by latlon

        labels_metadata : Pandas DataFrame / list / np array
                          labels metadata
        latlon : str
                 lat,lon

        returns : DataFrame
        """
        if len(labels_metadata) == 0:
            return pd.DataFrame()

        # Initialize dataframes
        labels_metadata = pd.DataFrame(labels_metadata)

        lat,lon = latlon.split(',')
        label_metadata = labels_metadata[(labels_metadata.loc[:,3] == float(lat)) &
                                         (labels_metadata.loc[:,2] == float(lon))]

        return label_metadata

    def get_timerange(self, label_metadata):
        """
        Get first and last time from label metadata

        label_metadata : DataFrame
                         label metadata (assumed to be in order)

        returns : DateTime, DateTime
                  starttime, endtime
        """

        label_metadata = label_metadata.sort_values(0)
        start = dt.datetime.fromtimestamp(int(label_metadata.iloc[0,1]))
        end = dt.datetime.fromtimestamp(int(label_metadata.iloc[-1,1]))
        return start,end

    def filter_ground_obs(self, obs, label_metadata):
        """
        Filter ground observations so that only train stations where trains
        have actually visited during current every hour are kept

        obs : list
              observations got from SmartMet Server
        label_metadata : list
                         labels metadata

        returns : DataFrame, DataFrame
                  metadata, data
        """
        if len(obs) == 0 or len(label_metadata) == 0:
            return [], []

        logging.debug('Filtering ground observations...')

        # Initialize dataframes
        labels_metadata = pd.DataFrame(label_metadata)

        # Take only observations from stations where have been train
        filt_labels_metadata, filt_obs = self._filter_obs(obs, label_metadata)

        #result = result.drop(columns=[0,1])

        filt_obs.fillna(-99, inplace=True)

        logging.debug("Obs shape: {} | Filt obs shape: {}".format(obs.shape, filt_obs.shape))
        logging.debug("Labels length: {} | Filt labels shape: {}".format(len(label_metadata), filt_labels_metadata.shape))

        return filt_labels_metadata, filt_obs

    def filter_flashes(self, flash_df, data_df):
        """
        Filter flashes

        flash_df : DataFrame
                   flashes
        data_df : DataFrame
                  data frame where data is appended to

        returns : DataFrame
                  data with appended values
        """
        if len(flash_df) == 0:
            data_df['flash'] = 0
            return data_df

        data_df = data_df.sort_values('time')

        for h, values in data_df.iterrows():
            count = len(flash_df[flash_df.time <= values.time])
            data_df.loc[h, 'flash'] = count
            flash_df = flash_df[flash_df.time > values.time]

        return data_df

    def filter_precipitation(self, data, prec_column=5):
        """
        Filter ground observations so that only train stations where trains
        have actually visited during current every hour are kept

        data : DataFrame
               DataFrame where labels are appended to

        returns : DataFrame
                 data
        """

        # Go through data and calculate 3h and 6h sums
        data = self._calc_prec_sums(data, prec_column).iloc[6:]
        return data


    def find_best_station(self, obs_df):
        """
        Find best station from dataframe

        obs_df : DataFrame
                 DataFrame formed by for example get_ground_obs

        returns : DataFrame
        """
        if len(obs_df) == 0:
            return obs_df

        bestrow = obs_df.apply(lambda x: x.count(), axis=1).idxmax()
        beststation = obs_df.loc[bestrow,0]

        obs_df = obs_df[(obs_df.loc[:,0] == beststation)]
        obs_df = obs_df.drop(columns=[0])

        return obs_df

    def sort_columns_by_list(self, df, cols):
        """
        Sort columns based on given list

        df : DataFrame
             data frame
        cols : list
               list of column names

        returns : DataFrame
        """

        pass

    def sort_by_latlon(self, df, lat_column='lat', lon_column='lon'):
        """
        Sort pandas dataframe by location

        df : DataFrame
             pandas dataframe
        lat_column : str
                     column name where latitude is stored
        lon_column : str
                     column name where longitude is stored

        returns : DataFrame
        """

        return df.assign(f = df[lat_column] + df[lon_column]).sort_values(by=['f', 'time']).drop('f', axis=1)


    def _filter_obs(self, obs, labels_df):
        """
        Filter observations based on metadata (private method)
        """
        # Create comparison hashes
        labels_df.loc[:,1] = labels_df.loc[:,1].astype(int)
        obs.loc[:,'time'] = obs.loc[:,'time'].astype(int)

        # Mask observations
        obs_mask = np.isin(obs.loc[:,'time'], labels_df.loc[:,1])
        filt_obs = obs[(obs_mask)]

        # Filter labels metadata
        labels_mask = np.isin(labels_df.loc[:,1], obs.loc[:,'time'])
        filt_labels_metadata = labels_df[(labels_mask)]

        return filt_labels_metadata, filt_obs

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

    def df_to_serving_json(self, df):
        """
        Convert pandas dataframe to serving json

        df : DataFrame
             dataframe to convert

        return str
        """

        df.drop(columns=['time','place'], inplace=True)
        js = '{"instances": ['
        for h,values in df.iterrows():
            js += '{"X": ['+','.join(list(map(str,values)))+']},'
        js += ']}'

        return js


    def df_to_serving_file(self, df):
        """
        Write observations to files (100 lines per each) required by
        gcloud cmd sdk

        df : Pandas DataFrame
             observations

        return : list
                 files
        """
        pred_data = df.drop(columns=['time','place', 'origintime'])
        files = []

        tmp = tempfile.NamedTemporaryFile(delete=False)
        i = 0
        f = open(tmp.name, 'w')

        #pred_data = data.drop(columns=['origintime', 'time', 'place'])
        for h,values in pred_data.iterrows():
            i += 1

            f.write('{"X": ['+','.join(list(map(str, values)))+']}\n')
            if i > 99:
                files.append(tmp.name)
                tmp = tempfile.NamedTemporaryFile(delete=False)
                f.close()
                f = open(tmp.name, 'w')
                i = 0

        if tmp.name not in files:
            files.append(tmp.name)

        return files

    def predict_gcloud_ml(self, model, version, files, weather_data, names):

        """
        Predict delays using gcloud command line sdk

        model : str
                model name
        version : str
                  model version
        files : list
                list of files where observations are located
        weather_data : Pandas DataFrame
              weather_dataervations, used to map prediction to latlons
        names : dict
                station names in format {latlon: name, ...}

        returns : dict
                  result in format {name: [(time, lat, lon, delay), ...], ...}
        """
        ret = []

        for x_file in files:
            cmd = 'gcloud --quiet ml-engine predict --model {model} --version {version} --json-instances {x_file}'.format(model=model, version=version, x_file=x_file)

            logging.debug('Using cmd: {}'.format(cmd))
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            lines = result.stdout.decode('utf-8').split('\n')

            for line in lines[1:-1]:
                ret.append(line[1:-1])

        ret = self._prediction_to_locations(ret, weather_data, names)
        return ret

    def predict_json(self, project, model, instances, version=None):
        """
        NOT WORKING in environments in hand

        Send json data to a deployed model for prediction.

        Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
        Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
        """
        # Create the ML Engine service object.
        # To authenticate set the environment variable
        # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>

        service = googleapiclient.discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}'.format(project, model)

        if version is not None:
            name += '/versions/{}'.format(version)

        response = service.projects().predict(
            name=name,
            body={'instances': instances}
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response['predictions']

    def _prediction_to_locations(self, pred, weather_data, names):
        """
        Map prediction to locations

        pred : list
               list of predicted delays
        weather_data : Pandas DataFrame
              weather_data (used to map predictions to latlons
        names : dict
                station names in format {latlon: name, ...}

        returns : dict
                  result in format {name: [(time, lat, lon, delay), ...], ...}
        """
        res = {}
        for h,values in weather_data.iterrows():
            name = names[values.place]
            if name not in res:
                res[name] = []

            res[name].append((values.time, values.lat, values.lon, float(pred.pop(0)), values.origintime))

        return res
    #
    # SASSE
    #
    def get_classes(self, X, thresholds=[0, 0.2, 0.4, 0.6]):

        """
        Get classes from X based on thresholds.
        """
        classes = []
        for x in X:
            i=len(thresholds) - 1
            _class = i
            while i >= 0:
                if x <= thresholds[i]:
                    _class = i
                i -= 1
            classes.append(_class)
        return classes
