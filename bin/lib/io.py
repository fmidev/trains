# -*- coding: utf-8 -*-
import sys, re, tempfile
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.externals import joblib
from keras.models import Model, model_from_yaml
import boto3
import random
import pickle
import json

class IO:

    s3 = False
    gs = False
    bucket_name = ''
    bucket = ''
    client = ''
    
    def __init__(self, s3_bucket=False, gs_bucket=False):

        if s3_bucket != False:
            self.bucket_name = s3_bucket
            self.s3 = True
            self.client = boto3.client('s3')
            resource = boto3.resource('s3')
            self.bucket = resource.Bucket(self.bucket_name)
        elif gs_bucket != False:
            self.bucket_name = gs_bucket
            self.gs = True
            # TODO continue
    


            

    #
    # GENERAL
    #
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
            if name == loc[1]:
                return loc[0]

        return None    
    

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

