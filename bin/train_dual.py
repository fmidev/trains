import sys, os, argparse, logging, json, itertools
import datetime as dt
from datetime import timedelta
from configparser import ConfigParser

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score

from lib import bqhandler, imputer, config
import lib.transformer as _trans
from lib.io import IO
from lib.viz import Viz

def main():
    """
    Get data from db and save it as csv
    """
    # Helpers
    bq = bqhandler.BQHandler()
    io = IO(gs_bucket=options.gs_bucket)
    viz = Viz(io)

    # Configuration
    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    # Initialise classifier
    if hasattr(options, 'classifier_file'):
        classifier = io.load_scikit_model(options.classifier_file)
    else:
        if options.classifier == 'svc':
            classifier = SVC()
        elif options.classifier == 'rfc':
            classifier = RandomForestClassifier(n_jobs=-1)
        else:
            raise('Model not specificied or wrong. Add "classifier: bgm" to config file.')

    # Initialise regression model
    if options.model == 'rfr':
        model = RandomForestRegressor(n_estimators=options.n_estimators,
                                      n_jobs=-1,
                                      min_samples_leaf=options.min_samples_leaf,
                                      min_samples_split=options.min_samples_split,
                                      max_features=options.max_features,
                                      max_depth=options.max_depth,
                                      bootstrap=options.bootstrap
                                      )
        regressor = _trans.Regressor(model=model)
    else:
        raise('Model not specificied or wrong. Add "classifier: bgm" to config file.')

    # Initialise transformer
    transformer = _trans.Selector(classifier=classifier)

    # Initialise pipeline
    pipe = Pipeline(
        [('selector', transformer),
         ('regression', regressor)]
    )

    sum_columns = ['delay']
    if options.reason_code_table is not None:
        sum_columns = ['count']

    logging.info('Reading data...')
    bq.set_params(loc_col='trainstation',
                  project=options.project,
                  dataset=options.feature_dataset,
                  table=options.feature_table,
                  parameters=all_param_names,
                  reason_code_table=options.reason_code_table)

    data = bq.get_rows(starttime,
                       endtime)

    data = io.filter_train_type(labels_df=data,
                                train_types=options.train_types,
                                sum_types=True,
                                train_type_column='train_type',
                                location_column='trainstation',
                                time_column='time',
                                sum_columns=sum_columns,
                                aggs=aggs)

    data.sort_values(by=['time', 'trainstation'], inplace=True)

    logging.info('Processing {} rows...'.format(len(data)))

    # Binary class
    logging.info('Adding binary class to the dataset with limit {}...'.format(options.delay_limit))
    data['class'] = data['delay'].map(lambda x: 1 if x > options.delay_limit else -1)

    # Balance
    if options.balance:
        logging.info('Balancing data...')
        count = data.groupby('class').size().min()
        data = pd.concat([data.loc[data['class'] == -1].sample(n=count),
        data.loc[data['class'] == 1].sample(n=count)])

    io.log_class_dist(data.loc[:, 'class'].values, labels=[-1,1])

    # Adding month
    if options.month:
        logging.info('Adding month to the dataset...')
        data['month'] = data['time'].map(lambda x: x.month)
        options.feature_params.append('month')

    data.set_index('time', inplace=True)

    y = data.loc[:,['delay', 'class']].astype(np.int32).values
    X = data.loc[:,options.feature_params].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    io.log_class_dist(y_train[:,1], [-1,1])

    n_samples, n_dims = X_train.shape

    if options.normalize:
        logging.info('Normalizing data...')
        if hasattr(options, 'xscaler_file'):
            scaler = io.load_scikit_model(options.xscaler_file)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

    if options.cv:
        logging.info('Doing random search for hyper parameters...')
        raise("No param_grid set for given model ({})".format(options.model))

        random_search = RandomizedSearchCV(model,
                                           param_distributions=param_grid,
                                           n_iter=int(options.n_iter_search),
                                           n_jobs=-1)

        random_search.fit(X_train, y_train)
        logging.info("RandomizedSearchCV done.")
        fname = options.output_path+'/random_search_cv_results.txt'
        report_cv_results(random_search.cv_results_, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)
        sys.exit()
    else:
        logging.info('Training...')
        transformer.set_y(y_train[:,0])
        regressor.set_classifier(transformer)
        pipe.fit(X_train, y_train[:,1])

    # Metrics
    print(pipe.steps)
    y_pred = pipe.steps[0][1].predict(X_test, type='int')

    io.save_scikit_model(pipe, filename=options.save_file, ext_filename=options.save_file)

    # Classification performance
    y_test_class = y_test[:,1]

    acc = accuracy_score(y_test_class, y_pred)
    precision = precision_score(y_test_class, y_pred, average='micro')
    recall = recall_score(y_test_class, y_pred, average='micro')
    f1 = f1_score(y_test_class, y_pred, average='micro')

    logging.info('Classification accuracy: {}'.format(acc))
    logging.info('Classification precision: {}'.format(precision))
    logging.info('Classification recall: {}'.format(recall))
    logging.info('Classification F1 score: {}'.format(f1))
    io.log_class_dist(y_pred, [-1, 1])

    # Confusion matrices
    fname = '{}/confusion_matrix_validation.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(4), filename=fname)

    fname = '{}/confusion_matrix_validation_normalised.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(4), True, filename=fname)

    if options.normalize:
        fname=options.save_path+'/xscaler.pkl'
        io.save_scikit_model(scaler, filename=fname, ext_filename=fname)

    if options.model == 'rfr':
        fname = options.output_path+'/rfc_feature_importance.png'
        viz.rfc_feature_importance(pipe.steps[1][1].get_feature_importances(), fname, feature_names=options.feature_params)

    # Regression performance
    #y_pred = pipe.steps[1][1].predict(X_test)
    y_test_reg = y_test[:,0]
    pipe.set_params(selector__full=True)
    y_pred = pipe.predict(X_test, full=True)

    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)

    logging.info('Regression RMSE: {}'.format(rmse))
    logging.info('Regression MAE: {}'.format(mae))
    logging.info('Regression R2 score: {}'.format(r2))

    error_data = {'acc': [acc],
                  'precision': [precision],
                  'recall': [recall],
                  'f1': [f1],
                  'rmse': [rmse],
                  'mae': [mae],
                  'r2': [r2]}
    fname = '{}/training_time_classification_validation_errors.csv'.format(options.output_path)
    io.write_csv(error_data, filename=fname, ext_filename=fname)


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', type=str, default=None, help='Configuration file name')
    parser.add_argument('--config_name', type=str, default=None, help='Configuration file name')
    parser.add_argument('--dev', type=int, default=0, help='1 for development mode')

    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    options = parser.parse_args()

    config.read(options)

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    logging.info('Using configuration: {} | {}'.format(options.config_filename, options.config_name))

    main()
