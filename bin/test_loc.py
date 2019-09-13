import sys, os, math
import numpy as np
import pandas as pd

from lib.model_functions.LocalizedLasso import LocalizedLasso

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

label_params = ['delay']
feature_params = ['pressure','min_temperature','mean_winddirection','mean_windspeedms','max_windgust','min_vis','min_clhb','max_precipitation3h']
meta_params = ['trainstation','time','train_type','train_count']


def main():
    """
    Get data from db and save it as csv
    """

    filename = 'test_train.csv'
    print('Reading data from file {}...'.format(filename))

    data = pd.read_csv(filename)
    graph_data = pd.read_csv('data/full/a_b_2010-18/xaa', names=['date', 'start_hour', 'src', 'dst', 'type', 'sum_delay','sum_ahead','add_delay','add_ahead','train_count'])

    train, test = train_test_split(data, test_size=0.3)

    X_train = train.loc[:, feature_params].astype(np.float32).values
    y_train = train.loc[:, label_params].astype(np.float32).values.ravel()

    X_test = test.loc[:, feature_params].astype(np.float32).values
    y_test = test.loc[:, label_params].astype(np.float32).values.ravel()

    # graph_data = pd.read_csv(options.graph_data, names=['date', 'start_hour', 'src', 'dst', 'type', 'sum_delay','sum_ahead','add_delay','add_ahead','train_count'])
    # graph = model.fetch_connections(graph_data)

    print('Features shape after pre-processing: {}'.format(X_train.shape))

    # print('One batch...')
    # model = LocalizedLasso(num_iter=10, batch_size=None)
    # graph = model.fetch_connections(graph_data)
    # model.fit(X_train, y_train, train.loc[:, 'trainstation'].values)
    #
    # y_pred, _ = model.predict(X_test, test.loc[:, 'trainstation'].values)
    # print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)))
    #
    #
    #
    # print('Batches...')
    # model = LocalizedLasso(num_iter=10, batch_size=50)
    # graph = model.fetch_connections(graph_data)
    # model.fit(X_train, y_train, train.loc[:, 'trainstation'].values)
    #
    # y_pred, _ = model.predict(X_test, test.loc[:, 'trainstation'].values)
    # print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)))


    print('Partial fit...')
    model = LocalizedLasso(num_iter=10, batch_size=50)
    graph = model.fetch_connections(graph_data)

    batch_size = 50
    start = 0
    end = start + batch_size
    while start < len(X_train):
        X_batch = X_train[start:end, :]
        y_batch = y_train[start:end]
        stations = train.loc[:, 'trainstation'].values[start:end]
        model.partial_fit(X_batch, y_batch, stations)
        start += batch_size
        end += batch_size

    y_pred, _ = model.predict(X_test, test.loc[:, 'trainstation'].values)
    print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)))


if __name__=='__main__':
    main()
