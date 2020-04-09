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




class EmptyDataError(Exception):
   """Empty data exception"""
   pass

def perf_metrics(y_pred_proba, y_pred, y_test, start, end, viz, io):
    """ Calculate, print, save and plot performance metrics """

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    logging.info('Accuracy: {}'.format(acc))
    logging.info('Precision: {}'.format(precision))
    logging.info('Recall: {}'.format(recall))
    logging.info('F1 score: {}'.format(f1))
    io.log_class_dist(y_pred, labels=[-1,1])

    error_data = {'acc': [acc],
                  'precision': [precision],
                  'recall': [recall],
                  'f1': [f1]}
    fname = '{}/test_validation_errors_{}_{}.csv'.format(options.output_path, start, end)
    io.write_csv(error_data, filename=fname, ext_filename=fname)

    # Confusion matrices
    fname = '{}/confusion_matrix_testset_{}_{}.png'.format(options.output_path, start, end)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(2), filename=fname)

    fname = '{}/confusion_matrix_testset_{}_{}_normalised.png'.format(options.output_path, start, end)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(2), True, filename=fname)

    # Precision-recall curve
    fname = '{}/precision-recall-curve_testset_{}_{}.png'.format(options.output_path, start, end)
    viz.prec_rec_curve(y_test, y_pred_proba, filename=fname)

    # ROC
    fname = '{}/roc_testset_{}_{}.png'.format(options.output_path, start, end)
    viz.plot_binary_roc(y_test, y_pred_proba, filename=fname)


def predict_timerange(test_data, feature_params, model, xscaler, start, end):
    """ Do prediction for given time range in data."""
    X = test_data.loc[start:end, feature_params].astype(np.float32).values
    y = test_data.loc[start:end, 'class'].astype(np.int32).values.ravel()

    if X.shape[0] < 1:
        raise EmptyDataError

    if options.normalize:
        X = xscaler.transform(X)

    logging.info('Predicting for time range {} - {}...'.format(start, end))
    y_pred_proba = model.predict_proba(X)
    y_pred = np.argmax(y_pred_proba, axis=1)
    # We want [-1,1] classes as y values are
    y_pred[y_pred == 0] = -1
    logging.info('...done')

    return y_pred_proba, y_pred, y





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
            params = {'kernel': options.kernel, 'gamma': options.gamma, 'C': options.penalty, 'probability': options.probability}
            classifier = SVC(**params)
        elif options.classifier == 'rfc':
            classifier = RandomForestClassifier(n_jobs=-1)
        else:
            raise('Model not specificied or wrong. Add "classifier: bgm" to config file.')

    # Initialise regression model
    if options.regression == 'rfr':
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

    # Pick only selected month
    where = {}
    if options.pick_month is not None:
        where = {'EXTRACT(MONTH from time)': options.pick_month}

    logging.info('Reading data...')
    bq.set_params(loc_col='trainstation',
                  project=options.project,
                  dataset=options.feature_dataset,
                  table=options.feature_table,
                  parameters=all_param_names,
                  reason_code_table=options.reason_code_table,
                  where=where)

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

    # Filter only timesteps with large distribution in the whole network
    if options.filter_delay_limit is not None:
        data = io.filter_delay_with_limit(data, options.filter_delay_limit)

    # Binary class
    logging.info('Adding binary class to the dataset with limit {}...'.format(options.delay_limit))
    data['class'] = data['delay'].map(lambda x: 1 if x > options.delay_limit else -1)

    # Separate train and validation sets
    data_train, data_test = train_test_split(data, test_size=0.3)

    # Balance
    if options.balance:
        logging.info('Balancing training data...')
        count = data_train.groupby('class').size().min()
        # SVC can't handle more than 50 000 samples
        if options.classifier == 'svc': count = min(count, 50000)
        data_train = pd.concat([data_train.loc[data_train['class'] == -1].sample(n=count),
        data_train.loc[data_train['class'] == 1].sample(n=count)])

    logging.info('Train data:')
    io.log_class_dist(data_train.loc[:, 'class'].values, labels=[-1,1])
    logging.info('Test data:')
    io.log_class_dist(data_test.loc[:, 'class'].values, labels=[-1,1])

    # Adding month
    if options.month:
        logging.info('Adding month to the datasets...')
        data_train['month'] = data_train.loc[:,'time'].map(lambda x: x.month)
        data_test['month'] = data_test.loc[:,'time'].map(lambda x: x.month)
        options.feature_params.append('month')

    data_train.set_index('time', inplace=True)

    y_train = data_train.loc[:,['delay', 'class']].astype(np.int32).values
    y_test = data_test.loc[:,['delay', 'class']].astype(np.int32).values
    X_train = data_train.loc[:,options.feature_params].astype(np.float32).values
    X_test = data_test.loc[:,options.feature_params].astype(np.float32).values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # io.log_class_dist(y_train[:,1], [-1,1])

    # If asked, save used train and test splits into big query
    if options.save_data:
        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_train'
        bq.nparray_to_table([X_train, y_train],
                            [options.feature_params, ['delay', 'class']],
                            options.project,
                            options.feature_dataset,
                            tname
                            )
        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_test'
        bq.nparray_to_table([X_test, y_test],
                            [options.feature_params, ['delay', 'class']],
                            options.project,
                            options.feature_dataset,
                            tname
                            )

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
        raise("No param_grid set for given model ({})".format(options.regression))

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
    y_pred_proba = pipe.steps[0][1].predict_proba(X_test)
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
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(2), filename=fname)

    fname = '{}/confusion_matrix_validation_normalised.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(2), True, filename=fname)

    # Precision-recall curve
    fname = '{}/precision-recall-curve.png'.format(options.output_path)
    viz.prec_rec_curve(y_test_class, y_pred_proba, filename=fname)

    # ROC
    fname = '{}/roc.png'.format(options.output_path)
    viz.plot_binary_roc(y_test_class, y_pred_proba, filename=fname)


    if options.normalize:
        fname=options.save_path+'/xscaler.pkl'
        io.save_scikit_model(scaler, filename=fname, ext_filename=fname)

    if options.regression == 'rfr':
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






    ############################################################################
    # EVALUATE
    ############################################################################
    if options.evaluate:
        logging.info('Loading test data...')
        test_data = bq.get_rows(dt.datetime.strptime('2010-01-01', "%Y-%m-%d"),
                                dt.datetime.strptime('2019-01-01', "%Y-%m-%d"),
                                loc_col='trainstation',
                                project=options.project,
                                dataset=options.feature_dataset,
                                table=options.test_table,
                                parameters=all_param_names)

        test_data = io.filter_train_type(labels_df=test_data,
                                         train_types=['K','L'],
                                         sum_types=True,
                                         train_type_column='train_type',
                                         location_column='trainstation',
                                         time_column='time',
                                         sum_columns=['delay'],
                                         aggs=aggs)

        # Sorting is actually not necessary. It's been useful for debugging.
        test_data.sort_values(by=['time', 'trainstation'], inplace=True)
        test_data.set_index('time', inplace=True)
        logging.info('Test data contain {} rows...'.format(len(test_data)))

        logging.info('Adding binary class to the test dataset with limit {}...'.format(options.delay_limit))
        #logging.info('Adding binary class to the dataset with limit {}...'.format(limit))
        #data['class'] = data['count'].map(lambda x: 1 if x > options.delay_count_limit else -1)
        test_data['class'] = test_data['delay'].map(lambda x: 1 if x > options.delay_limit else -1)
        io.log_class_dist(test_data.loc[:, 'class'].values, labels=[-1,1])

        if options.month:
            logging.info('Adding month to the test dataset...')
            test_data['month'] = test_data.index.map(lambda x: x.month)

        times = [('2014-01-01', '2014-02-01'), ('2016-06-01', '2016-07-01'), ('2017-02-01', '2017-03-01'), ('2011-02-01', '2017-03-01')]
        for start, end in times:
            try:
                y_pred_proba, y_pred, y = predict_timerange(test_data, options.feature_params, pipe.steps[0][1], scaler, start, end)
                perf_metrics(y_pred_proba, y_pred, y, start, end, viz, io)
            except EmptyDataError:
                logging.info('No data for {} - {}'.format(start, end))


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
