from math import radians, sin, cos, asin, sqrt
import sys
import pandas as pd
import numpy as np
import logging
import multiprocessing

_X = pd.DataFrame()
_DISTANCE_LIMIT = 150
_SEARCH_SPAN = 5

def _apply_df(args):
    """
    Helper function for apply_by_multiprocessing
    """
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    """
    Do pandas apply in paraller
    """
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
                     for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

def fast_dist(lon1, lat1, lon2, lat2, val, ind):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    """
    if pd.isnull(val):
        return ind, np.NaN

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    R = 6371  #radius of the earth in km
    x = (lon2 - lon1) * cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    dist = R * sqrt( x*x + y*y )

    if dist > _DISTANCE_LIMIT:
        return ind, np.NaN

    return ind, dist

def closest(lat, lon, i, _time, _loc_name):
    """
    Provide closest non-missing value to given lat,lon coordinates and _time from global _X.
    Note that _X need to be arranged by lat to narrow the search to n nearest station

    lat : float
          latitude of point of interest
    lon : float
          longitude of point of interest
    i   : int
          we consider i:th column in the dataframe
    _time : str
            time of missing value
    _loc_name : Location name of missing value. Used to narrow search to n:th nearest locations

    return : value from nearest station or np.NaN if it can't be found
    """
    # pick time of interest from global dataframe
    X_t = _X.loc[_X['time'] == _time]

    # reset index to to pick n nearest location.
    X_t.reset_index(inplace=True)

    # pick n value around from given station
    index = X_t.index[X_t['trainstation'] == _loc_name].values[0]
    start = index - _SEARCH_SPAN
    if start < 0: start = 0
    end = index + _SEARCH_SPAN
    if end > len(X_t): end = len(X_t)
    X_t = X_t.loc[start:end, :]

    # Calculate distances to nearest stations. Drop missing distances
    # (missing distances appear, if there is no station where value is not
    # np.NaN in the n nearest station and inside given radius)
    distances = X_t.apply(lambda p: fast_dist(lon, lat, p['lon'], p['lat'], p[i], p['index']), axis=1, result_type='expand')
    distances.dropna(inplace=True)
    # Return value from given location and i:th column or np.NaN
    if len(distances) < 1:
        return np.NaN

    return _X.loc[int(distances[1].idxmin()), i]

def r(r):
    """
    Go through all columns with missing values in the row and find closest
    non-missing value for them
    """
    lat = r['lat']
    lon = r['lon']
    _time = r['time']
    _name = r['trainstation']
    ns = r[r.isna()]
    for i in ns.index:
        closest_value = closest(lat, lon, i, _time, _name)
        r[i] = closest_value

    return r

def fit_transform(X, missing_value=-99):
    """
    Impute all missing values in X with geographically closest non-missing values.

    Algorithm:
    1. Arrange data based on latitude
    2. Go through all rows with missing values (in paraller)
    3. Go through all cols with missing value
    4. Search for nearest trainstation with non-missing values. Search is done
       within _SEARCH_SPAN stations for perfomance reasons (arranged from south
       to north). Only stations inside _DISTANCE_LIMIT are allowed.
    5. If non-missing value is found, impute missing value with that,
       else drop whole row.

    X : pd.DataFrame
        Data to impute. Dataframe need to have columns
        'train_type', 'trainstation', 'time', 'lat' and 'lon'

    return pd.DataFrame
    """
    #imputer = Imputer(strategy=self.strat)
    global _X
    global _DISTANCE_LIMIT
    global _SEARCH_SPAN

    cpu_count = multiprocessing.cpu_count()

    _X = pd.DataFrame(X)
    _X.drop(columns=['train_type'], inplace=True)
    pd.set_option('display.max_columns', 500)
    _X.replace(missing_value, np.NaN, inplace=True)
    _X.sort_values(by=['lat', 'time'], inplace=True)

    orig_len = len(_X)

    nulls = _X[_X.isnull().any(axis=1)]
    non_nulls = _X.dropna()
    null_len = len(nulls)

    #print(nulls)

    nulls = apply_by_multiprocessing(nulls, r, axis=1, workers=cpu_count, result_type='broadcast')
    nulls.dropna(inplace=True)

    new_null_len = len(nulls)
    logging.info('Imputation done.\n Total rows: {} \n Rows with missing values: {} \n Imputed rows: {}\n Dropped rows: {}'.format(orig_len, null_len, new_null_len, int(null_len - new_null_len)))

    ret = pd.concat([nulls, non_nulls])
    ret.loc[:, 'train_type'] = None

    return ret
