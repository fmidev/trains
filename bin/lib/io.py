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


class IO:

    s3 = False
    bucket_name = ''
    bucket = ''
    client = ''
    
    def __init__(self, s3_bucket=False):

        if s3_bucket != False:
            self.bucket_name = s3_bucket
            self.s3 = True
            self.client = boto3.client('s3')
            resource = boto3.resource('s3')
            self.bucket = resource.Bucket(self.bucket_name)
    

    def save_model(self, model_filename, weights_filename, history_filename, model, history):
        """ Save model and weights into file """
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
        """ Load model and weights from file """

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
        
    def write_probabilities(self, filename, ids, data):
        """ Write results to file """

        file = open(filename, 'w')
        file.write("Sample_id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,Class_10\n")
        i = 0
        for line in data:
            l = [ '%.4f' % elem for elem in line ]
            file.write(str(ids[i]) + ',' + ','.join(l)+'\n')
            i += 1
            
        file.close()

    def add_dim(self, x, value=1):
        new = []
        for row in x:
            new_row = []
            for i in x:
                new_row.append([i,value])
            new.append(new_row)
        return new

    def shuffle(self, X, y):
        print("Shuffling data...")
        c = list(zip(X, y))
        random.shuffle(c)
        X, y = zip(*c)
        X, y = list(X), list(y)
        return X, y

    
    def read_data(self, xfilename, yfilename=None,
                  data_type=float, delimiter=';',
                  skip_cols=0, skip_rows=1, force_local=False):
        """ Read data from files """
        X, y = [], []
        
        # Read data    
        if self.s3 and not force_local:
            tmp = tempfile.NamedTemporaryFile()
            self.bucket.download_file(xfilename, tmp.name)
            with open(tmp.name) as f:                
                lines = f.read().splitlines()
        else:
            with open(xfilename) as f:
                lines = f.read().splitlines()
            
        for line in lines[skip_rows:]:
            l = line.split(delimiter)[skip_cols:]
            X.append(list(map(data_type, l)))
        
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
    
    def split_data(self, x, y, num_of_batches, num_of_samples=None):
        """ Split data to train and validation data set 
        x:               [matrix, n,k]  feature list (samples as rows, features as cols)
        y:               [matrix, n,1]  label list
        num_of_batches:  [int]        number of batches to make
        num_of_samples:  [int]        number of samplest to pick to every batch (if None, all are taken)
        """

        selected, ids, batches_x, batches_y = [], [], [], []
        r = random.SystemRandom()
        k = len(x)
        batch_size = k/num_of_batches
        if num_of_samples is not None:
            if batch_size > num_of_samples:
                batch_size = num_of_samples
                
        for batch_num in range(num_of_batches):
            batch_x, batch_y, batch_ids = [], [], []
        
            while len(batch_x) < batch_size and len(x) > 0:
                i = r.randrange(0, len(x))            
                batch_x.append(x.pop(i))
                batch_y.append(y.pop(i))
                batch_ids.append(i)
            
            batches_x.append(np.matrix(batch_x))
            batches_y.append(np.matrix(batch_y).T)
            ids.append(batch_ids)
            
        return ids, batches_x, batches_y

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
