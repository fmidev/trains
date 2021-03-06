import sys, os, argparse, logging, json, itertools, copy
import datetime as dt
from datetime import timedelta
from configparser import ConfigParser

import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, r2_score, classification_report

from lib import bqhandler, imputer, config
import lib.transformer as _trans
from lib.io import IO
from lib.viz import Viz
from lib.state import State
from lib.gpclassifier import GPClassifier
from lib.nbclassifier import NBClassifier
from lib.gp import GP

NEG_CLASS = 0

class EmptyDataError(Exception):
   """Empty data exception"""
   pass

def report_cv_results(results, filename=None, n_top=3):
    res = ""
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            res += "Model with rank: {0}\n".format(i)
            res += "Mean validation score: {0:.3f} (std: {1:.3f})\n".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate])
            res += "Parameters: {0}\n".format(results['params'][candidate])
            res += "\n"

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(res)

    logging.info(res)

def perf_metrics(y_pred_proba, y_pred, y_test, name_suffix, viz, io, output_path):
    """ Calculate, print, save and plot performance metrics """

    if not os.path.exists(output_path): os.makedirs(output_path)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    logging.info('Accuracy: {}'.format(acc))
    logging.info('Precision: {}'.format(precision))
    logging.info('Recall: {}'.format(recall))
    logging.info('F1 score: {}'.format(f1))
    io.log_class_dist(y_pred, labels=[NEG_CLASS,1])

    error_data = {'acc': [acc],
                  'precision': [precision],
                  'recall': [recall],
                  'f1': [f1]}
    fname = '{}/test_validation_errors_{}.csv'.format(output_path, name_suffix)
    io.write_csv(error_data, filename=fname, ext_filename=fname)

    # Confusion matrices
    fname = '{}/confusion_matrix_testset_{}.png'.format(output_path, name_suffix)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(2), filename=fname)

    fname = '{}/confusion_matrix_testset_{}_normalised.png'.format(output_path, name_suffix)
    viz.plot_confusion_matrix(y_test, y_pred, np.arange(2), True, filename=fname)

    # Precision-recall curve
    fname = '{}/precision-recall-curve_testset_{}.png'.format(output_path, name_suffix)
    viz.prec_rec_curve(y_test, y_pred_proba, filename=fname)

    # ROC
    fname = '{}/roc_testset_{}.png'.format(output_path, name_suffix)
    viz.plot_binary_roc(y_test, y_pred_proba, filename=fname)


def predict_timerange(test_data, feature_params, models, scalers, start, end):
    """
    Do prediction for given time range in data.

    So far only classifier implemented

    test_data      : DataFrame
                     Data containing X and y
    feature_params : tuple
                     (classifier_feature_params, regressor_feature_params)
    models         : tuple
                     (classifier, regressor)
    scalers        : tuple
                     (classifier_xscaler, regressor_scaler, regressor_yscaler)
    start          : DateTime
                     period start time
    end            : DateTime
                     period end time

    Return y_pred_proba, y_pred, y
    """
    X = test_data.loc[start:end, feature_params[0]].astype(np.float32).values
    y = test_data.loc[start:end, 'class'].astype(np.int32).values.ravel()

    if X.shape[0] < 1:
        raise EmptyDataError

    if options.normalize_classifier:
        X = scalers[0].transform(X)

    logging.info('Predicting for time range {} - {} ({} rows)...'.format(start, end, len(X)))

    y_pred = models[0].predict(X)
    y_pred_proba = models[0].predict_proba(X)
    #y_pred = np.argmax(y_pred_proba, axis=1)

    # Ensure that classes are correct
    y_pred[y_pred == 0] = NEG_CLASS
    logging.info('...done')

    return y_pred_proba, y_pred, y





def main(location=None):
    """
    ...
    """
    if location is None:
        locations = options.locations
        location = 'all'
    else:
        locations = [location]

    logging.info('Processing {}...'.format(location))
    output_path = options.output_path+'/{}'.format(location)
    save_path = options.save_path+'/{}'.format(location)

    if not os.path.exists(output_path): os.makedirs(output_path)
    if not os.path.exists(save_path): os.makedirs(save_path)


    ############################################################################
    # FETCHING AND PROCESSING DATA
    ############################################################################

    all_param_names = list(set(options.label_params + options.feature_params + options.meta_params + options.classifier_feature_params + options.regressor_feature_params))

    # Param list is modified after retrieving data
    classifier_feature_params = copy.deepcopy(options.classifier_feature_params)
    regressor_feature_params = copy.deepcopy(options.regressor_feature_params)

    all_feature_params = list(set(options.feature_params + options.meta_params + options.classifier_feature_params + options.regressor_feature_params))
    aggs = io.get_aggs_from_param_names(all_feature_params)

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
                  locations=locations,
                  reason_code_table=options.reason_code_table,
                  reason_codes_exclude=options.reason_codes_exclude,
                  reason_codes_include=options.reason_codes_include,
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

    if data.shape[0] < 1:
        logging.warning('Empty dataset for {}'.format(location))
        return
    else:
        logging.info('Processing {} rows...'.format(len(data)))

    # Filter only timesteps with large distribution in the whole network
    if options.filter_delay_limit is not None:
        data = io.filter_delay_with_limit(data, options.filter_delay_limit)

    # Binary class
    logging.info('Adding binary class to the dataset with limit {}...'.format(options.delay_limit))
    data['class'] = data['delay'].map(lambda x: 1 if x > options.delay_limit else NEG_CLASS)

    # Separate train and validation sets
    data_train, data_test = train_test_split(data, test_size=0.1)

    # Balance
    if options.balance:
        try:
            logging.info('Balancing training data...')
            count = data_train.groupby('class').size().min()
            if options.n_samples is not None:
                count = int(min(count, options.n_samples/(options.balance_ratio+1)))

            count2 = int(min((options.balance_ratio*count), data_train.loc[(data_train['class'] == 0)].shape[0]))

            data_train = pd.concat([data_train.loc[data_train['class'] == NEG_CLASS].sample(n=count2),
            data_train.loc[data_train['class'] == 1].sample(n=count)])
        except:
            logging.warning('No samples in some class:\n{}',format(data_train.groupby('class').size().min()))
            return


    logging.info('Train data:')
    io.log_class_dist(data_train.loc[:, 'class'].values, labels=[NEG_CLASS,1])
    logging.info('Test data:')
    io.log_class_dist(data_test.loc[:, 'class'].values, labels=[NEG_CLASS,1])

    # SVC can't handle more than 50 000 samples
    # GNB performs better with smaller training set
    if options.classifier == 'svc':
        data_train = data_train.sample(min(50000, len(data_train)))
    if options.classifier == 'bayes':
        data_train = data_train.sample(min(10000, len(data_train)))

    if len(data_train) < 300:
        logging.warning('Too few training samples ({} limit being {}). Skippting location {}'.format(len(data_train), 300, location))
        return

    # Adding month
    if options.month:
        logging.info('Adding month to the datasets...')
        data_train = data_train.assign(month=lambda df: df.loc[:, 'time'].map(lambda x: x.month))
        data_test = data_test.assign(month=lambda df: df.loc[:, 'time'].map(lambda x: x.month))
        regressor_feature_params.append('month')
        classifier_feature_params.append('month')
        #options.feature_params.append('month')

    data_train.set_index('time', inplace=True)

    y_train = data_train.loc[:,['delay', 'class']].astype(np.int32).values
    y_test = data_test.loc[:,['delay', 'class']].astype(np.int32).values
    X_train_classifier = data_train.loc[:,classifier_feature_params].copy().astype(np.float32).values
    X_test_classifier = data_test.loc[:,classifier_feature_params].copy().astype(np.float32).values
    X_train_regressor = data_train.loc[:,regressor_feature_params].copy().astype(np.float32).values
    X_test_regressor = data_test.loc[:,regressor_feature_params].copy().astype(np.float32).values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # io.log_class_dist(y_train[:,1], [-1,1])

    # If asked, save used train and test splits into big query
    if options.save_data:
        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_train'
        tname = tname.replace('-','_')
        df = data_train.copy()
        df['stationname'] = location
        bq.dataset_to_table(df, options.feature_dataset, tname)

        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_test'
        tname = tname.replace('-','_')
        df = data_test.copy()
        df['stationname'] = location
        bq.dataset_to_table(df, options.feature_dataset, tname)




    ############################################################################
    # NORMALIZE
    ############################################################################
    if options.normalize_classifier:
        logging.info('Normalizing classifier data...')
        if hasattr(options, 'xscaler_file_classifier'):
            state.xscaler_classifier = io.load_scikit_model(options.xscaler_file_classifier)
            X_train_classifier = state.xscaler_classifier.transform(X_train_classifier)
        else:
            xscaler_classifier = StandardScaler()
            X_train_classifier = xscaler_classifier.fit_transform(X_train_classifier)
            # Save classifier
            fname=options.save_path+'/xscaler_classifier.pkl'
            io.save_scikit_model(xscaler_classifier, filename=fname, ext_filename=fname)
            state.xscaler_classifier = xscaler_classifier

        X_test_classifier = state.xscaler_classifier.transform(X_test_classifier)
        #state.SCALER_CLASSIFIER_FITTED = True

    if options.normalize_regressor:
        logging.info('Normalizing regression data...')
        if hasattr(options, 'xscaler_file_regressor'):
            state.xscaler_regressor = io.load_scikit_model(options.xscaler_file_regressor)
            X_train_regressor = state.xscaler_regressor.transform(X_train_regressor)
        else:
            xscaler_regressor = StandardScaler()
            X_train_regressor = xscaler_regressor.fit_transform(X_train_regressor)
            # Save
            fname=options.save_path+'/xscaler_regressor.pkl'
            io.save_scikit_model(xscaler_regressor, filename=fname, ext_filename=fname)
            state.xscaler_regressor = xscaler_regressor

        X_test_regressor = state.xscaler_regressor.transform(X_test_regressor)

        if hasattr(options, 'yscaler_file_regressor'):
            state.yscaler_regressor = io.load_scikit_model(options.yscaler_file_regressor)
            y_train[:, 0] = state.yscaler_regressor.transform(y_train[:, 0].reshape(-1, 1)).ravel()
        else:
            yscaler_regressor = StandardScaler()
            #print(y_train[:,0].reshape(-1,1).shape)
            y_train[:, 0] = yscaler_regressor.fit_transform(y_train[:,0].reshape(-1,1)).ravel()
            fname=options.save_path+'/yscaler_regressor.pkl'
            io.save_scikit_model(yscaler_regressor, filename=fname, ext_filename=fname)
            state.yscaler_regressor = yscaler_regressor

        #state.SCALER_REGRESSOR_FITTED = True




    ############################################################################
    # MODEL INITIALIZATION
    ############################################################################

    # Initialise classifier
    if hasattr(options, 'classifier_file'):
        fname = options.classifier_model_file.replace('{location}', location)
        classifier = io.load_scikit_model(fname)
    else:
        if options.classifier == 'svc':
            params = {'kernel': options.kernel, 'gamma': options.gamma, 'C': options.penalty, 'probability': options.probability}
            classifier = SVC(**params)
        elif options.classifier == 'rfc':
            classifier = RandomForestClassifier(
                n_estimators=options.n_estimators,
                n_jobs=-1,
                min_samples_leaf=options.min_samples_leaf,
                min_samples_split=options.min_samples_split,
                max_features=options.max_features,
                max_depth=options.max_depth,
                bootstrap=options.bootstrap
                )
        elif options.classifier == 'gp':
            #classifier = GPClassifier(noise_level=options.noise_level)
            M = min(1000, X_train_classifier.shape[0])
            Z = X_train_classifier[:M, :].copy()
            classifier = GP(dim=X_train_classifier.shape[1], save_path=options.save_path, output_path=options.output_path) #, Z=Z)
            if options.restore:
                classifier.load(io)
        elif options.classifier == 'gpscikit':
            classifier = GPClassifier(noise_level=options.noise_level)
        elif options.classifier == 'bayes':
            classifier = NBClassifier()
        else:
            raise('Model not specificied or wrong. Add "classifier: gp" to config file.')

    # Initialise regression model
    if hasattr(options, 'regressor_file'):
        fname = options.regressor_file.replace('{location}', location)
        regressor = io.load_scikit_model(fname)
        regressor.fitted = True
    elif hasattr(options, 'regressor_model_file'):
        fname = options.regressor_model_file.replace('{location}', location)
        model = io.load_scikit_model(fname)
        regressor = _trans.Regressor(model=model, fitted=True)
    elif options.regression == 'rfr':
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
    transformer = _trans.Selector(classifier=classifier, regressor=regressor)
    regressor.set_classifier(transformer)

    # Initialise pipeline
    # pipe = Pipeline(
    #     [('selector', transformer),
    #      ('regression', regressor)]
    # )







    ############################################################################
    # TRAINING
    ############################################################################

    if options.cv:
        # Only RFC implemented
        logging.info('Doing random search for hyper parameters for classifier...')

        if options.classifier == 'rfc':
            param_grid = {"n_estimators": [5, 10, 25, 50, 100, 200],
                          "max_depth": [3, 10, 20, None],
                          "max_features": ["auto", "sqrt", "log2", None],
                          "min_samples_split": [2,5,10],
                          "min_samples_leaf": [1, 2, 4, 10],
                          "bootstrap": [True, False]}
        else:
            raise("No param_grid set for given model ({})".format(options.regression))

        random_search = RandomizedSearchCV(classifier,
                                           param_distributions=param_grid,
                                           scoring='f1_macro',
                                           n_iter=int(options.n_iter_search),
                                           n_jobs=-1)

        random_search.fit(X_train_classifier, y_train[:,1])
        logging.info("RandomizedSearchCV done.")
        fname = options.output_path+'/random_search_cv_classifier_results.txt'.format(location)
        report_cv_results(random_search.cv_results_, fname)
        io._upload_to_bucket(filename=fname, ext_filename=fname)
        sys.exit()
    else:
        # Training
        if not hasattr(options, 'classifier_file'): # TODO do we have fitted argument?
            logging.info('Training classifier...')

            try:
                classifier.fit(X_train_classifier, y_train[:,1])
            except LinAlgError as e:
                logging.warning(e)
                return

            # Save classifier
            fname = save_path+'/classifier.pkl'
            if options.classifier == 'gp':
                plot_filename = output_path+'/train_elbo.png'
                viz.plot_elbo(classifier.logf, 10, classifier.n_iter, plot_filename)
                classifier.save(fname, io)
            else:
                io.save_scikit_model(classifier, filename=fname, ext_filename=fname)

            # Show training perfomance
            y_pred = classifier.predict(X_train_classifier)
            y_pred = y_pred[~np.isnan(y_pred)]
            y_ = y_train[~np.isnan(y_pred)][:,1]
            logging.info("Training classification performance: \n{}".format(classification_report(y_, y_pred)))

            state.add_pred(classifier.predict_proba(X_train_classifier)[~np.isnan(y_pred)], y_pred, y_, 'train')
            if options.plot_learning:
                fname = output_path+'/learning_curve.png'
                viz.plot_learning_curve(classifier, X_train_classifier, y_train[:,1], filename=fname)

            if options.classifier == 'bayes':
                fname = output_path+'/model_features_on_gaussian.png'
                viz.plot_gnb_features(classifier.model, options.classifier_feature_params, filename=fname)

        if not regressor.fitted:
            logging.info('Training regressor...')
            regressor.fit(X_train_regressor, y_train[:,0])

            # Save classifier
            fname = save_path+'/regressor.pkl'
            io.save_scikit_model(model, filename=fname, ext_filename=fname)

            # Show training performance
            y_pred = regressor.predict(X_train_regressor, X_train_classifier)
            rmse = np.sqrt(mean_squared_error(y_train[:,0], y_pred))
            mae = mean_absolute_error(y_train[:,0], y_pred)
            r2 = r2_score(y_train[:,0], y_pred)

            logging.info('Training regression RMSE: {}'.format(rmse))
            logging.info('Training regression MAE: {}'.format(mae))
            logging.info('Training regression R2 score: {}'.format(r2))


        # Pipe approach, only class 1 samples used in regressor training
        #transformer.set_y(y_train[:,0])
        #pipe.fit(X_train, y_train[:,1])
        # Save pipe
        # io.save_scikit_model(pipe.steps[1][1], filename=fname, ext_filename=fname)
        # io.save_scikit_model(pipe, filename=options.save_file, ext_filename=options.save_file)









    ############################################################################
    # EVALUATE WITH VALIDATION DATASET
    ############################################################################

    # Metrics
    y_pred = classifier.predict(X_test_classifier)
    y_pred_proba = classifier.predict_proba(X_test_classifier)

    # Preidction may contain NaN because gpclassifier end up to erroro in/ (2 * np.sqrt(var_f_star * 2 * np.pi))
    mask = np.any(np.isnan(y_pred_proba), axis=1)
    y_pred = y_pred[~mask]
    y_pred_proba = y_pred_proba[~mask]

    # y_pred_proba = pipe.steps[0][1].predict_proba(X_test)
    # y_pred = pipe.steps[0][1].predict(X_test, type='int')

    # Classification performance
    y_test_class = y_test[~mask][:,1]

    state.add_pred(y_pred_proba, y_pred, y_test_class, 'validation')

    acc = accuracy_score(y_test_class, y_pred)
    precision = precision_score(y_test_class, y_pred, average='micro')
    recall = recall_score(y_test_class, y_pred, average='micro')
    f1 = f1_score(y_test_class, y_pred, average='micro')

    logging.info('Classification accuracy: {}'.format(acc))
    logging.info('Classification precision: {}'.format(precision))
    logging.info('Classification recall: {}'.format(recall))
    logging.info('Classification F1 score: {}'.format(f1))
    io.log_class_dist(y_pred, [NEG_CLASS, 1])

    # Confusion matrices
    fname = '{}/confusion_matrix_validation.png'.format(output_path)
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(2), filename=fname)

    fname = '{}/confusion_matrix_validation_normalised.png'.format(output_path)
    viz.plot_confusion_matrix(y_test_class, y_pred, np.arange(2), True, filename=fname)

    # Precision-recall curve
    fname = '{}/precision-recall-curve.png'.format(output_path)
    viz.prec_rec_curve(y_test_class, y_pred_proba, filename=fname)

    # ROC
    fname = '{}/roc.png'.format(output_path)
    viz.plot_binary_roc(y_test_class, y_pred_proba, filename=fname)


    # if options.regression == 'rfr':
    #     fname = output_path+'/rfr_feature_importance.png'
    #     viz.rfc_feature_importance(regressor.get_feature_importances(), fname, feature_names=regressor_feature_params)
    #     #viz.rfc_feature_importance(pipe.steps[1][1].get_feature_importances(), fname, feature_names=options.feature_params)

    # Regression performance
    #y_pred = pipe.steps[1][1].predict(X_test)
    y_test_reg = y_test[:,0]

    #pipe.set_params(selector__full=True)
    #y_pred = pipe.predict(X_test, full=True)
    y_pred = regressor.predict(X_test_regressor, X_test_classifier)

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
    fname = '{}/training_time_classification_validation_errors.csv'.format(output_path)
    io.write_csv(error_data, filename=fname, ext_filename=fname)






    ############################################################################
    # EVALUATE WITH TESTSET
    ############################################################################
    if options.evaluate:
        logging.info('Loading test data...')
        s = dt.datetime.strptime('2010-01-01', "%Y-%m-%d")
        e = dt.datetime.strptime('2019-01-31', "%Y-%m-%d")

        reason_code_table, reason_codes_exclude, reason_codes_include = None, None, None
        #if not options.skip_evaluation_reason_codes:
        reason_code_table=options.reason_code_table
        reason_codes_exclude=options.reason_codes_exclude
        reason_codes_include=options.reason_codes_include

        test_data = bq.get_rows(s,
                                e,
                                loc_col='trainstation',
                                project=options.project,
                                dataset=options.feature_dataset,
                                table=options.test_table,
                                locations=options.locations,
                                parameters=all_param_names,
                                reason_code_table=reason_code_table,
                                reason_codes_exclude=reason_codes_exclude,
                                reason_codes_include=reason_codes_include
                                )

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
        if test_data.shape[0] < 1:
            logging.warning('Test data contain 0 rows')
            return
        else:
            logging.info('Test data contain {} rows...'.format(len(test_data)))

        logging.info('Adding binary class to the test dataset with limit {}...'.format(options.delay_limit))
        #logging.info('Adding binary class to the dataset with limit {}...'.format(limit))
        #data['class'] = data['count'].map(lambda x: 1 if x > options.delay_count_limit else -1)
        test_data['class'] = test_data['delay'].map(lambda x: 1 if x > options.delay_limit else NEG_CLASS)
        io.log_class_dist(test_data.loc[:, 'class'].values, labels=[NEG_CLASS,1])

        if options.month:
            logging.info('Adding month to the test dataset...')
            test_data = test_data.assign(month=lambda df: df.index.map(lambda x: x.month))
            test_data['month'] = test_data.index.map(lambda x: x.month)

        #times = [('2014-01-01', '2014-02-01'), ('2016-06-01', '2016-07-01'), ('2017-02-01', '2017-03-01'), ('2011-02-01', '2017-03-01')]
        #times = [('2011-01-01', '2011-03-01'), ('2016-06-01', '2016-07-01'), ('2017-02-01', '2017-03-01'), ('2011-02-01', '2017-03-01')]
        #times = [('2010-01-01', '2010-01-31')]
        if options.test_times is not None:
            times = options.test_times
        else:
            times = [('2011-01-01', '2011-03-01'), ('2016-06-01', '2016-07-01'), ('2017-02-01', '2017-03-01')]

        for start, end in times:
            try:
                td = test_data
                if location is not None and location != 'all':
                    td = test_data[(test_data['trainstation'] == location)]
                y_pred_proba, y_pred, y = predict_timerange(td,
                                                            (classifier_feature_params, regressor_feature_params),
                                                            (classifier, regressor),
                                                            (state.xscaler_classifier, state.xscaler_regressor, state.yscaler_regressor),
                                                            start,
                                                            end)
                perf_metrics(y_pred_proba, y_pred, y, '{}-{}'.format(start, end), viz, io, output_path)
                state.add_pred(y_pred_proba, y_pred, y, 'test-{}-{}'.format(start, end))
            except EmptyDataError:
                logging.info('No data for {} - {}'.format(start, end))


        # Whole timerange
        try:
            start = times[0][0]
            end = times[-1][-1]
            y_pred_proba, y_pred, y = predict_timerange(test_data,
                                                        (classifier_feature_params, regressor_feature_params),
                                                        (classifier, regressor),
                                                        (state.xscaler_classifier, state.xscaler_regressor, state.yscaler_regressor),
                                                        start,
                                                        end)
            perf_metrics(y_pred_proba, y_pred, y, '{}-{}'.format(start, end), viz, io, output_path)
            state.add_pred(y_pred_proba, y_pred, y, 'test-{}-{}'.format(start, end))

        except EmptyDataError:
            logging.info('No data for {} - {}'.format(start, end))

    logging.info('done\n--------------------------------------------------------')


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', type=str, default=None, help='Configuration file name')
    parser.add_argument('--config_name', type=str, default=None, help='Configuration file name')
    parser.add_argument('--dev', type=int, default=0, help='1 for development mode')
    parser.add_argument('--skip_evaluation_reason_codes', dest='skip_evaluation_reason_codes', action='store_true', help='If set, reason codes are ignored in evaluation')
    parser.set_defaults(skip_evaluation_reason_codes=False)


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

    # Helpers
    bq = bqhandler.BQHandler()
    io = IO(gs_bucket=options.gs_bucket)
    viz = Viz(io)
    state = State()

    starttime, endtime = io.get_dates(options)
    logging.info('Using dataset {} and time range {} - {}'.format(options.feature_dataset,
                                                                  starttime.strftime('%Y-%m-%d'),
                                                                  endtime.strftime('%Y-%m-%d')))


    if options.save_data:
        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_train'
        tname = tname.replace('-','_')
        bq.delete_table(options.project, options.feature_dataset, tname)
        tname = options.model+'_'+options.feature_dataset+'_'+options.config_name+'_test'
        tname = tname.replace('-','_')
        bq.delete_table(options.project, options.feature_dataset, tname)

    if (options.station_specific_classifier or options.station_specific_regressor) and options.locations is not None:
        for location in options.locations:
            main(location)
    else:
        main(None)

    for name in state.get_pred_names():
        y_pred_proba_all, y_pred_all, y_all = state.get_pred(name)
        logging.info('Aggregated performance metrics for {}:'.format(name))
        perf_metrics(y_pred_proba_all, y_pred_all, y_all, name, viz, io, options.output_path+'/all')
        logging.info('................................................................\n')
