import sys, os, argparse, logging, json, itertools
import datetime as dt
from datetime import timedelta
from configparser import ConfigParser

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score

from lib import bqhandler, imputer, config
import lib.transformer as _trans
from lib.io import IO
from lib.viz import Viz
from lib.convlstm import LSTMClassifier
from lib.modelloader import ModelLoader
from lib.predictor import Predictor, PredictionError
from lib.svcclassifier import SVCClassifier
from lib.gaussiannbclassifier import GaussianNBClassifier
from lib.graphsvcclassifier import GraphSVCClassifier

class EmptyDataError(Exception):
   """Empty data exception"""
   pass

binary_labels = [0,1]

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
    io.log_class_dist(y_pred, labels=binary_labels)

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
    y_pred = model.predict(X)
    y_pred_proba = model.y_pred_proba
    #y_pred_proba = model.predict_proba(X)
    #y_pred = np.argmax(y_pred_proba, axis=1)

    # We want [-1,1] classes as y values are
    #y_pred[y_pred == 0] = 0

    # LSTM may not first time steps
    y_ = y[(len(y)-len(y_pred_proba)):]

    logging.info('...done')

    return y_pred_proba, y_pred, y_





def main():
    """
    Get data from db and save it as csv
    """
    # Helpers
    bq = bqhandler.BQHandler()
    io = IO(gs_bucket=options.gs_bucket)
    viz = Viz(io)
    predictor = Predictor(io, ModelLoader(io), options)

    ### OPTIONS ################################################################

    # Configuration
    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))

    all_param_names = options.label_params + options.feature_params + options.meta_params
    aggs = io.get_aggs_from_param_names(options.feature_params)

    ### MODELS #################################################################

    # Initialise classifier
    if hasattr(options, 'classifier_file'):
        classifier = io.load_scikit_model(options.classifier_file)
    else:
        if options.classifier == 'svc':
            params = {'kernel': options.kernel, 'gamma': options.gamma, 'C': options.penalty, 'probability': options.probability}
            #classifier = SVC(**params)
            classifier = SVCClassifier(params, limit=options.class_limit)
        elif options.classifier == 'graphsvc':

            classifier  = GraphSVCClassifier()
            graph_data = pd.read_csv(options.graph_data, names=['date', 'start_hour', 'src', 'dst', 'type', 'sum_delay','sum_ahead','add_delay','add_ahead','train_count'])
            classifier.fetch_connections(graph_data)

        elif options.classifier == 'gaussiannb':
            classifier = GaussianNBClassifier()
        elif options.classifier == 'lstm':
            num_of_features = len(options.feature_params)
            if options.month: num_of_features += 1
            class_weight=None
            if hasattr(options, 'class_weight'):
                class_weight=eval(options.class_weight)
            params = {'length': options.time_steps, 'batch_size': options.batch_size, 'epochs': options.epochs, 'num_of_features': num_of_features, 'log_dir': options.log_dir, 'class_weight':class_weight}
            classifier = LSTMClassifier(**params)
        else:
            raise('Model not specificied or wrong. Add "classifier: bgm" to config file.')

    # Initialise regression model
    if options.regression == 'rfr':
        regressor = RandomForestRegressor(n_estimators=options.n_estimators,
                                      n_jobs=-1,
                                      min_samples_leaf=options.min_samples_leaf,
                                      min_samples_split=options.min_samples_split,
                                      max_features=options.max_features,
                                      max_depth=options.max_depth,
                                      bootstrap=options.bootstrap
                                      )
        #regressor = _trans.Regressor(model=model)
    else:
        raise('Model not specificied or wrong. Add "classifier: bgm" to config file.')

    # Initialise transformer
    #transformer = _trans.Selector(classifier=classifier)

    # Initialise pipeline
    #pipe = Pipeline(
    #    [('selector', transformer),
    #     ('regression', regressor)]
    #)

    ### DATA ###################################################################

    sum_columns = ['delay']
    if 'train_count' in options.meta_params:
        sum_columns.append('train_count')

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
                  locations=options.locations,
                  only_winters=options.only_winters,
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

    data['delay'] = data.loc[:, 'delay'].replace(-99, np.nan)
    data.sort_values(by=['trainstation', 'time'], inplace=True)

    logging.info('Processing {} rows...'.format(len(data)))

    # Filter only timesteps with large distribution in the whole network
    if options.filter_delay_limit is not None:
        data = io.filter_delay_with_limit(data, options.filter_delay_limit)

    # Binary class
    logging.info('Adding binary class to the dataset with limit {}...'.format(options.delay_limit))
    def set_class(x):
        if x > options.delay_limit:
            return binary_labels[1]
        elif x < options.delay_limit:
            return binary_labels[0]
        return np.nan
    data['class'] = data['delay'].map(lambda x: set_class(x))

    # Separate train and validation sets
    data_train, data_test = train_test_split(data, test_size=0.3, shuffle=False)

    # Balance
    if options.balance:
        logging.info('Balancing training data...')
        count = data_train.groupby('class').size().min()
        # SVC can't handle more than 50 000 samples
        if options.classifier == 'svc': count = min(count, 50000)
        data_train = pd.concat([data_train.loc[data_train['class'] == 0].sample(n=count),
        data_train.loc[data_train['class'] == 1].sample(n=count)])

    logging.info('Train data:')
    io.log_class_dist(data_train.loc[:, 'class'].values, labels=binary_labels)
    logging.info('Test data:')
    io.log_class_dist(data_test.loc[:, 'class'].values, labels=binary_labels)

    # Adding month
    if options.month:
        logging.info('Adding month to the datasets...')
        data_train['month'] = data_train.loc[:,'time'].map(lambda x: x.month)
        data_test['month'] = data_test.loc[:,'time'].map(lambda x: x.month)
        options.feature_params.append('month')

    #data_train.set_index('time', inplace=True)
    #y_train_class = data_train.loc[:,['class']].astype(np.int32).values.ravel()
    #y_train_delay = data_train.loc[:,['delay']].astype(np.int32).values.ravel()
    y_train_class = data_train.loc[:,['class']].values.ravel()
    y_train_delay = data_train.loc[:,['delay']].values.ravel()
    #y_test_class = data_test.loc[:,['class']].astype(np.int32).values.ravel()
    #y_test_delay = data_test.loc[:,['delay']].astype(np.int32).values.ravel()
    y_test_class = data_test.loc[:,['class']].values.ravel()
    y_test_delay = data_test.loc[:,['delay']].values.ravel()

    X_train = data_train.loc[:,options.feature_params].astype(np.float32).values
    X_test = data_test.loc[:,options.feature_params].astype(np.float32).values

    if options.smote:
        logging.info('Smoting...')
        sm = SMOTE()
        X_train_class, y_class = sm.fit_resample(X_train, y_train_class)
        io.log_class_dist(y_class, labels=binary_labels)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # io.log_class_dist(y_train[:,1], [-1,1])

    # If asked, save used train and test splits into big query
    if options.save_data:
        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_train'
        columns = [options.feature_params, ['delay'], ['class']]
        bq.nparray_to_table([X_train, y_train_class, y_train_delay],
                            columns,
                            options.project,
                            options.feature_dataset,
                            tname
                            )
        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_test'
        bq.nparray_to_table([X_test, y_test_class, y_test_delay],
                            columns,
                            options.project,
                            options.feature_dataset,
                            tname
                            )

    if options.normalize:
        logging.info('Normalizing data...')
        #scale=(0,1)
        if hasattr(options, 'xscaler_file'):
            xscaler = io.load_scikit_model(options.xscaler_file)
            X_train = xscaler.transform(X_train)
            X_test = xscaler.transform(X_test)
        else:
            xscaler = MinMaxScaler(feature_range=(-1,1))
            #xscaler = StandardScaler()
            X_train = xscaler.fit_transform(X_train)
            X_test = xscaler.transform(X_test)
            fname = options.save_path+'/xscaler.pkl'
            io.save_scikit_model(xscaler, fname, fname)

        if hasattr(options, 'yscaler_file'):
            yscaler = io.load_scikit_model(options.yscaler_file)
            y_train_delay = yscaler.transform(y_train_delay)
            y_test_delay = yscaler.transform(y_test_delay)
        else:
            #yscaler = MinMaxScaler(feature_range=(0,1))
            yscaler=StandardScaler()
            y_train_delay = yscaler.fit_transform(y_train_delay.reshape(-1,1)).ravel()
            y_test_delay = yscaler.transform(y_test_delay.reshape(-1,1)).ravel()
            fname = options.save_path+'/yscaler.pkl'
            io.save_scikit_model(yscaler, fname, fname)


    data_train.loc[:,options.feature_params].to_csv('data/x_train.csv', index=False)
    data_test.loc[:,options.feature_params].to_csv('data/x_test.csv', index=False)
    data_train.loc[:,['class']].fillna(-99).astype(np.int).to_csv('data/y_train.csv', index=False)
    data_test.loc[:,['class']].fillna(-99).astype(np.int).to_csv('data/y_test.csv', index=False)
    sys.exit()

    ### TRAIN ##################################################################

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
        logging.info('Training classifier...')

        if options.classifier == 'graphsvc':
            classifier.fit(X_train, y_train_class, stations=data_train.loc[:, 'trainstation'].values)
        else:
            history = classifier.fit(X_train, y_train_class, X_test, y_test_class)

        # Save classifier
        if options.classifier == 'lstm':
            history_fname = options.save_path+'/history.pkl'
            fname = options.save_path+'/classifier.h5'
            io.save_keras_model(fname, history_fname, classifier, history.history)
        else:
            fname = options.save_path+'/classifier.pkl'
            io.save_scikit_model(classifier, filename=fname, ext_filename=fname)

        # Drop data with no delay information
        X_train = X_train[~np.isnan(y_train_delay)]
        y_train_delay = y_train_delay[~np.isnan(y_train_delay)]
        y_train_class = y_train_class[~np.isnan(y_train_class)]

        y_pred_train_bin = classifier.predict(X_train, type='bool')

        # debug
        #y_pred_train_bin
        #indices = np.random.choice(np.arange(y_pred_train_bin.size),
        #                           replace=False,
        #                           size=int(y_pred_train_bin.size * 0.2))
        #y_pred_train_bin[indices] = True

        #print('y_pred_train_bin: {}'.format(y_pred_train_bin.shape))
        #print('y_train_delay: {}'.format(y_train_delay.shape))
        #print('y_train_class: {}'.format(y_train_class.shape))

        # Pick only severe values
        #y_train_delay_ = y_train_delay[(len(y_train_class)-len(y_pred_train_bin)):]
        #X_train_ = X_train[(len(y_train_class)-len(y_pred_train_bin)):]
        y_train_delay_ = y_train_delay[(len(y_train_delay)-len(y_pred_train_bin)):]
        X_train_ = X_train[(len(y_train_delay)-len(y_pred_train_bin)):]
        #print('y_train_delay_: {}'.format(y_train_delay_.shape))
        y_train_severe = y_train_delay_[y_pred_train_bin]
        X_train_severe = X_train_[y_pred_train_bin]

        logging.info('Training regressor...')
        regressor.fit(X_train_severe, y_train_severe)

    # Save regressor
    io.save_scikit_model(regressor, filename=options.save_file, ext_filename=options.save_file)

    # Learning history
    # fname = options.output_path+'/learning_over_time.png'
    # viz.plot_nn_perf(history.history, metrics={'Error': {'mean_squared_error': 'MSE',
    #                                                      'mean_absolute_error': 'MAE'}},
    #                                                      filename=fname)

    ### RESULTS FOR VALIDATION SET #############################################

    # Drop data with missing delay
    X_test = X_test[~np.isnan(y_test_class)]
    y_test_class = y_test_class[~np.isnan(y_test_class)]
    data_test = data_test[~np.isnan(data_test.delay)]

    # Metrics
    #y_pred_proba = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.y_pred_proba

    #y_test_delay = y_test_delay[~np.isnan(y_test_delay)]

    # Classification performance
    # LSTM don't have first timesteps
    y_test_class = y_test_class[(len(X_test)-len(y_pred)):]

    acc = accuracy_score(y_test_class, y_pred)
    precision = precision_score(y_test_class, y_pred, average='micro')
    recall = recall_score(y_test_class, y_pred, average='micro')
    f1 = f1_score(y_test_class, y_pred, average='micro')

    logging.info('Classification accuracy: {}'.format(acc))
    logging.info('Classification precision: {}'.format(precision))
    logging.info('Classification recall: {}'.format(recall))
    logging.info('Classification F1 score: {}'.format(f1))
    io.log_class_dist(y_pred, binary_labels)

    # Confusion matrices
    fname = '{}/confusion_matrix_validation.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(2), filename=fname)

    fname = '{}/confusion_matrix_validation_normalised.png'.format(options.output_path)
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(2), True, filename=fname)

    # Precision-recall curve
    fname = '{}/precision-recall-curve_validation.png'.format(options.output_path)
    viz.prec_rec_curve(y_test_class, y_pred_proba, filename=fname)

    # ROC
    fname = '{}/roc_validation.png'.format(options.output_path)
    viz.plot_binary_roc(y_test_class, y_pred_proba, filename=fname)

    if options.regression == 'rfr':
        fname = options.output_path+'/rfc_feature_importance.png'
        viz.rfc_feature_importance(regressor.feature_importances_, fname, feature_names=options.feature_params)

    # Regression performance
    y_pred_reg, y_test_reg = predictor.pred(data=data_test)
    #y_test_reg = y_test[(len(y_test)-len(y_pred)):,0]
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
                                reason_code_table=options.reason_code_table,
                                locations=options.locations,
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
        #test_data.sort_values(by=['time', 'trainstation'], inplace=True)

        # Filter only timesteps with large distribution in the whole network
        if options.filter_delay_limit is not None:
            test_data = io.filter_delay_with_limit(test_data, options.filter_delay_limit)

        test_data.set_index('time', inplace=True)
        logging.info('Test data contain {} rows...'.format(len(test_data)))

        logging.info('Adding binary class to the test dataset with limit {}...'.format(options.delay_limit))
        test_data['class'] = test_data['delay'].map(lambda x: binary_labels[1] if x > options.delay_limit else binary_labels[0])
        io.log_class_dist(test_data.loc[:, 'class'].values, labels=binary_labels)

        if options.month:
            logging.info('Adding month to the test dataset...')
            test_data['month'] = test_data.index.map(lambda x: x.month)

        times = [('2014-01-01', '2014-02-01'), ('2016-06-01', '2016-07-01'), ('2017-02-01', '2017-03-01'), ('2011-02-01', '2011-03-01')]
        for start, end in times:
            try:
                y_pred_proba, y_pred, y = predict_timerange(test_data, options.feature_params, classifier, xscaler, start, end)
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
