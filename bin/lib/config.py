import os
from configparser import ConfigParser

def read(options):

    def _path(name, root_dir):
        ''' Read path from options and create it if not exists'''
        val = getattr(options, name, None)
        if val is None or val == 'None':
            val = root_dir+'/'+options.model+'/'+options.feature_dataset+'/'+options.config_name

        if not os.path.exists(val):
            os.makedirs(val)

        setattr(options, name, val)

    def _fval(name, default=None):
        ''' Convert float val to float taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = float(val)
        else:
            val = default
        setattr(options, name, val)

    def _bval(name, default=False):
        ''' Convert option from int to bool'''
        val = getattr(options, name, False)
        if int(val) == 1: val = True
        else: val = default
        setattr(options, name, val)

    def _intval(name, default=None):
        ''' Convert int val to integer taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = int(val)
        else:
            val = default
        setattr(options, name, val)


    parser = ConfigParser()
    parser.read(options.config_filename)

    if parser.has_section(options.config_name):
        params = parser.items(options.config_name)
        for param in params:
            if getattr(options, param[0], None) is None:
                setattr(options, param[0], param[1])

        options.feature_params = getattr(options, 'feature_params', None)
        if options.feature_params is not None:
            options.feature_params = options.feature_params.split(',')
        else:
            options.feature_params = []
        options.classifier_feature_params = getattr(options, 'classifier_feature_params', None)
        if options.classifier_feature_params is not None:
            options.classifier_feature_params = options.classifier_feature_params.split(',')
        else:
            options.classifier_feature_params = []
        options.regressor_feature_params = getattr(options, 'regressor_feature_params', None)
        if options.regressor_feature_params is not None:
            options.regressor_feature_params = options.regressor_feature_params.split(',')
        else:
            options.regressor_feature_params = []

        options.label_params = options.label_params.split(',')
        options.meta_params = options.meta_params.split(',')

        options.locations = getattr(options, 'locations', None)
        if options.locations is not None:
            options.locations = options.locations.split(',')

        try:
            options.gmm_params = options.gmm_params.split(',')
        except:
            pass

        try:
            periods = options.test_times.split('|')
            options.test_times = []
            for p in periods:
                options.test_times.append(p.split(','))
        except:
            options.test_times = None

        try:
            options.train_types = options.train_types.split(',')
        except:
            options.train_types = ['K', 'L']

        try:
            options.train_stations = options.train_stations.split(',')
        except:
            options.train_stations = None

        try:
            options.model_type = options.model_type
        except:
            options.model_type = 'scikit'

        _path('save_path', 'models')
        if options.model_type == 'keras':
            options.save_file = options.save_path+'/model.h5'
        else:
            options.save_file = options.save_path+'/model.pkl'

        _path('output_path', 'results')
        options.vis_path = options.output_path+'/validation_testset'
        if not os.path.exists(options.vis_path):
            os.makedirs(options.vis_path)

        try:
            _path('log_dir', '/board')
        except PermissionError:
            pass

        if not os.path.exists(options.vis_path):
                os.makedirs(options.vis_path)

        # common / several
        _bval('cv')
        _bval('pca')
        _bval('whiten')
        _bval('normalize')
        _bval('normalize_regressor')
        _bval('normalize_classifier')
        _bval('impute')
        _intval('n_loops')
        _intval('batch_size')
        _bval('tf')
        _intval('y_avg_hours')
        _intval('day_step', 5000)
        _intval('hour_step', 0)
        _bval('month')
        _bval('only_winters')
        _intval('epochs', 3)
        _bval('balance')
        _bval('evaluate')
        _bval('y_avg')
        _bval('save_data')
        _intval('pick_month')
        _intval('filter_delay_limit')
        _fval('class_limit', .5)
        _bval('smote')
        _fval('balance_ratio', 1)

        # GP
        _fval('noise_level', 5)

        try:
            if options.label_column is None:
                options.label_column = 'delay'
        except:
            options.label_column = 'delay'

        if not hasattr(options, 'reason_code_table'):
            options.reason_code_table = None

        if hasattr(options, 'reason_codes_exclude'):
            options.reason_codes_exclude = options.reason_codes_exclude.split(',')
        else:
            options.reason_codes_exclude = None

        if hasattr(options, 'reason_codes_include'):
            options.reason_codes_include = options.reason_codes_include.split(',')
        else:
            options.reason_codes_include = None

        # linear regression
        _fval('alpha')
        _fval('eta0')
        _fval('power_t')
        _bval('shuffle')

        # ard model
        _fval('alpha_1')
        _fval('alpha_2')
        _fval('lambda_1')
        _fval('lambda_2')
        _bval('fit_intercept')
        _bval('copy_X')
        _fval('threshold_lambda')
        _intval('n_samples')

        # rf
        _bval('bootstrap')
        _intval('n_estimators')
        _intval('min_samples_split')
        _intval('min_samples_leaf')
        _intval('max_depth')

        # SVC
        _bval('probability')
        _fval('penalty')
        _fval('gamma')

        # lstm
        _intval('time_steps')
        _intval('cell_size')
        _fval('lr')
        _intval('n_hidden')
        _fval('quantile')
        _fval('p_drop')
        _bval('slow')

        # kmeans
        _bval('kmeans')
        _intval('n_clusters')

        # BayesianGaussianMixture (bgm)
        _intval('n_components', 4)

        # binary
        _intval('delay_count_limit')
        _intval('delay_limit')

        # other
        _intval('pca_components')
        _intval('dbscan')
        _bval('feature_selection')


        return options
    else:
        raise Exception('Section {} not found in the {} file'.format(options.config_name, options.config_filename))

    return tables
