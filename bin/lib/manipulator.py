# -*- coding: utf-8 -*-
import sys, os
from sklearn.externals import joblib
import json
import logging

class Manipulator:

    def __init__(self):
        pass


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

        return joblib.load(str(filename))
