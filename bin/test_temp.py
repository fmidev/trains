import sys, os, math
import numpy as np
import pandas as pd

from lib.model_functions.NetworkLasso import NetworkLasso

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def main():
    """
    Get data from db and save it as csv
    """

    #model = NetworkLasso(num_iter=1000, mode='regression', report_interval=100)
    fname_x = 'temperatures.csv'
    fname_y = 'true_temperatures.csv'
    fname_g = 'A.csv'
    print('Reading data from files {} and {} and graph from file {}...'.format(fname_x, fname_y, fname_g))

    x_train = pd.read_csv(fname_x, header=None)
    y_train = pd.read_csv(fname_y, header=None)
    A = pd.read_csv(fname_g, header=None)

    print('x_train: {}'.format(x_train.shape))
    print('y_train: {}'.format(y_train.shape))
    print('A: {}'.format(A.shape))

    num_iter = 1000


    print('Fit in one batch...')
    model = NetworkLasso(num_iter=num_iter, batch_size=None)
    model.set_graph(A.values)

    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_train.values)

    print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_train.values, y_pred)), mean_absolute_error(y_train.values, y_pred)))




    print('Fit in batch larger than data...')
    model = NetworkLasso(num_iter=num_iter, batch_size=1024)
    model.set_graph(A.values)

    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_train.values)

    print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_train.values, y_pred)), mean_absolute_error(y_train.values, y_pred)))



    print('Fit in batches...')
    model = NetworkLasso(num_iter=num_iter, batch_size=20)
    model.set_graph(A.values)

    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_train.values)

    print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_train.values, y_pred)), mean_absolute_error(y_train.values, y_pred)))



    print('Fit in batches 2...')
    model = NetworkLasso(num_iter=num_iter, batch_size=41)
    model.set_graph(A.values)

    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_train.values)

    print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_train.values, y_pred)), mean_absolute_error(y_train.values, y_pred)))

    # So far partial fit can't be tested without station data
    # print('Partial fit in one batch...')
    # model = NetworkLasso(num_iter=num_iter, mode='network', batch_size=None)
    # model.set_graph(A.values)
    #
    # model.partial_fit(x_train.values, y_train.values)
    # y_pred = model.predict(x_train.values)
    #
    # print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_train.values, y_pred)), mean_absolute_error(y_train.values, y_pred)))
    #
    #
    #
    #
    # print('Partial fit with larger batch than data...')
    # model = NetworkLasso(num_iter=num_iter, mode='network', batch_size=1024)
    # model.set_graph(A.values)
    #
    # model.partial_fit(x_train.values, y_train.values)
    # y_pred = model.predict(x_train.values)
    #
    # print('RMSE: {}, MAE: {}'.format(np.sqrt(mean_squared_error(y_train.values, y_pred)), mean_absolute_error(y_train.values, y_pred)))


if __name__=='__main__':
    main()
