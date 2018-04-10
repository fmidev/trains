# -*- coding: utf-8 -*-
import sys, re, tempfile
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.externals import joblib
from keras.models import Model, model_from_yaml
from google.cloud import storage
import boto3
import random
import pickle
import json
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

    #
    # GENERAL
    #
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


    def read_parameters(self, filename, force_local=False):
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

        return lines

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
    def get_train_stations(self, filename=None):

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
       
        for s in data:
            latlon = {'lat': s['latitude'], 'lon': s['longitude']}
            stations[s['stationShortCode'].encode('utf-8').decode()] = latlon

        return stations

    def find_id(self, locations, name):
        """
        Find id from (id, name) tuple list
        """
        for loc in locations:
            if str(name) == str(loc[1]):
                return loc[0]

        raise KeyError('Id for name {} not found'.format(name))

    def filter_labels(self, labels_metadata, labels,
                      features_metadata, features,
                      invert=False, uniq=False):
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

        if len(features_metadata) == 0 or len(features) == 0:
            return labels_metadata, labels
        
        labels_metadata = np.array(labels_metadata)        
        
        if uniq:
            logging.debug('Dropping duplicate values...')
            df = pd.DataFrame(labels_metadata)
            mask = np.invert(df.duplicated([0,1]).as_matrix())
            labels_metadata = labels_metadata[(mask)]
            labels = labels[(mask)]
            
            logging.debug('Shape of uniq metadata: {}'.format(labels_metadata.shape))
            logging.debug('Shape of uniq data: {}'.format(labels.shape))
        
        mask = np.isin(labels_metadata, features_metadata)
        if invert:
            filtered_labels = labels[np.invert((mask[:,0] & mask[:,1]))]
            filtered_labels_metadata = labels_metadata[np.invert((mask[:,0] & mask[:,1]))]
        else:
            filtered_labels = labels[(mask[:,0] & mask[:,1])]
            filtered_labels_metadata = labels_metadata[(mask[:,0] & mask[:,1])]

        logging.debug('Shape of filtered data: {}'.format(filtered_labels.shape))
        logging.debug('Shape of filtered metadata: {}'.format(filtered_labels_metadata.shape))                                             

        return filtered_labels_metadata, filtered_labels

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

        label_metadata.sort_values(0, inplace=True)
        start = dt.datetime.fromtimestamp(int(label_metadata.iloc[0,0]))
        end = dt.datetime.fromtimestamp(int(label_metadata.iloc[-1,0]))
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
            data_df.loc[:,'flash'] = 0
            return data_df

        data_df = data_df.sort_values('time')

        for h, values in data_df.iterrows():
            count = len(flash_df[flash_df.time <= values.time])
            data_df.loc[h, 'flash'] = count
            flash_df = flash_df[flash_df.time > values.time]

        return data_df        
    
    def filter_precipitation(self, obs, data):
        """
        Filter ground observations so that only train stations where trains
        have actually visited during current every hour are kept
        
        obs : DataFrame
               observations got from SmartMet Server
        data : DataFrame
               DataFrame where labels are appended to 
        
        returns : DataFrame
                 data
        """

        if len(obs) == 0:
            data.loc[:,'3hsum'] = -99
            data.loc[:,'6hsum'] = -99
            return data

        # Go through data and calculate 3h and 6h sums
        obs = self._calc_prec_sums(obs)

        # Select correct vaues from obserations to every station and hour        
        data.set_index('time', drop=False, inplace=True)
        obs.loc[:,'time'] = obs.loc[:,'time'].astype(int)
        obs.set_index('time', drop=False, inplace=True)        

        result = pd.concat([data, obs.loc[:,['3hsum','6hsum']]], axis=1, join_axes=[data.index])
        # print(obs)
        # notfound = 0
        # for h,values in data.iterrows():
        #     try:
        #         data.loc[h,'3hsum'] = obs.loc[h,'3hsum']
        #         data.loc[h,'6hsum'] = obs.loc[h,'6hsum']
        #     except KeyError:
        #         notfound += 1            

        # logging.debug('Prec sum is missing for {} (out of {}) time-points'.format(notfound, len(data)))
        logging.debug("Obs shape: {} | result shape: {}".format(obs.shape, data.shape))

        return result
    
    
    def find_best_station(self, obs_df):
        """
        Find best station from dataframe
        
        obs_df : DataFrame
                 DataFrame formed by for example get_ground_obs
        
        returns : DataFrame
        """
        bestrow = obs_df.apply(lambda x: x.count(), axis=1).idxmax()
        beststation = obs_df.loc[bestrow,0]

        obs_df = obs_df[(obs_df.loc[:,0] == beststation)]
        obs_df = obs_df.drop(columns=[0])
    
        return obs_df
        
    
    def _filter_obs(self, obs, labels_df):
        """
        Filter observations based on metadata (private method)       
        """                
        # Create comparison hashes
        labels_df.loc[:,0] = labels_df.loc[:,0].astype({0: int})
        obs.loc[:,'time'] = obs.loc[:,'time'].astype(int)    

        # Mask observations
        obs_mask = np.isin(obs.loc[:,'time'], labels_df.loc[:,0])
        filt_obs = obs[(obs_mask)]

        # Filter labels metadata 
        labels_mask = np.isin(labels_df.loc[:,0], obs.loc[:,'time'])
        filt_labels_metadata = labels_df[(labels_mask)]
        
        return filt_labels_metadata, filt_obs

    def _calc_prec_sums(self, obs):
        """
        Calculate 3h and 6h prec sums (private method)
        """
        sum_3h = []
        sum_6h = []

        obs.loc[:,'3hsum'] = -99
        obs.loc[:,'6hsum'] = -99
        
        for h,values in obs.iterrows():
            sum_3h.append(values[5])
            sum_6h.append(values[5])

            if len(sum_3h) > 3:
                sum_3h.pop(0)
                obs.loc[h, '3hsum'] = sum(sum_3h)
                
            if len(sum_6h) > 6:
                sum_6h.pop(0)
                obs.loc[h, '6hsum'] = sum(sum_6h)

        return obs
    
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

