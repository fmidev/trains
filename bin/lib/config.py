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

    def _fval(name):
        ''' Convert float val to float taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = float(val)
        else:
            val = None
        setattr(options, name, val)

    def _bval(name):
        ''' Convert option from int to bool'''
        val = getattr(options, name, False)
        if int(val) == 1: val = True
        else: val = False
        setattr(options, name, val)

    def _intval(name):
        ''' Convert int val to integer taking possible None value into account'''
        val = getattr(options, name, None)
        if val is not None and val != 'None':
            val = int(val)
        else:
            val = None
        setattr(options, name, val)

    parser = ConfigParser()
    parser.read(options.config_filename)

    if parser.has_section(options.config_name):
        params = parser.items(options.config_name)
        for param in params:
            if getattr(options, param[0], None) is None:
                setattr(options, param[0], param[1])

        options.feature_params = options.feature_params.split(',')
        options.label_params = options.label_params.split(',')
        options.meta_params = options.meta_params.split(',')

        try:
            options.train_types = options.train_types.split(',')
        except:
            options.train_types = ['K', 'L']

        _path('save_path', 'models')
        _path('output_path', 'results')
        options.vis_path = options.output_path+'/validation_testset'
        if not os.path.exists(options.vis_path):
            os.makedirs(options.vis_path)

        _path('log_dir', '/tmp')
        options.save_file = options.save_path+'/model.pkl'

        # common / several
        _bval('cv')
        _bval('pca')
        _bval('whiten')
        _bval('normalize')
        _bval('impute')
        _intval('n_loops')
        _intval('batch_size')
        _bval('tf')

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

        # lstm
        _intval('time_steps')
        _intval('cell_size')
        _fval('lr')
        _intval('n_hidden')
        _fval('quantile')
        _fval('p_drop')

        # kmeans
        _bval('kmeans')
        _intval('n_clusters')

        # other
        _intval('pca_components')
        _intval('dbscan')

        return options
    else:
        raise Exception('Section {} not found in the {} file'.format(options.config_name, options.config_filename))

    return tables
