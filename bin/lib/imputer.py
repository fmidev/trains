from math import radians, sin, cos, asin, sqrt
import sys
import pandas as pd
import numpy as np
import logging
import multiprocessing

_X = pd.DataFrame()

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
                     for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

def fast_dist(lon1, lat1, lon2, lat2, val):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    """
    if val is None:
        return np.NaN
    R = 6371  #radius of the earth in km
    x = (lon2 - lon1) * cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    return R * sqrt( x*x + y*y )


def closest(lat, lon, i, _time, _loc_name):
    X_t = _X.loc[_X['time'] == _time]
    X_t.reset_index(drop=True, inplace=True)
    index = X_t.index[X_t['trainstation'] == _loc_name].values[0]
    span = 2
    start = index - span
    if start < 0: start = 0
    end = index + span
    if end > len(X_t): end = len(X_t)
    X_t = X_t.loc[start:end, :]
    distances = X_t.apply(lambda p: fast_dist(lon, lat, p['lon'], p['lat'], p[i]), axis=1)
    distances.dropna(inplace=True)
    if len(distances) < 1:
        return np.NaN
    return _X.loc[distances.idxmin(), i]

def r(r):
    lat = r['lat']
    lon = r['lon']
    _time = r['time']
    _name = r['trainstation']
    ns = r[r.isna()]
    for i in ns.index:
        # logging.debug('None value: {}'.format(i))
        closest_value = closest(lat, lon, i, _time, _name)
        r[i] = closest_value

    return r

def fit_transform(X):
    """
    Impute all missing values in X that meet the user specified threshold.
    """
    #imputer = Imputer(strategy=self.strat)
    global _X
    _X = pd.DataFrame(X)
    _X.drop(columns=['train_type'], inplace=True)
    pd.set_option('display.max_columns', 500)
    _X.replace(-99.0, np.NaN, inplace=True)
    _X.sort_values(by=['lat', 'time'], inplace=True)

    orig_len = len(_X)

    #print(X)

    nulls = _X[_X.isnull().any(axis=1)]
    non_nulls = _X.dropna()
    null_len = len(nulls)

    #print(nulls)

    nulls = apply_by_multiprocessing(nulls, r, axis=1, workers=multiprocessing.cpu_count(), result_type='broadcast')
    nulls.dropna(inplace=True)

    new_null_len = len(nulls)
    logging.info('Imputation done.\n Total rows: {} \n Rows with missing values: {} \n Imputed rows: {}\n Dropped rows: {}'.format(orig_len, null_len, new_null_len, int(null_len - new_null_len)))

    ret = pd.concat([nulls, non_nulls])
    ret.loc[:, 'train_type'] = None

    return ret
