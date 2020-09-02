import sys, os, argparse, logging, json

# import tensorflow as tf
# from tensorflow.python.estimator.export import export
# from tensorflow.python.framework import constant_op
# #from tensorflow.contrib import predictor
#
# from tensorflow.keras.models import load_model

from sklearn import metrics


# from lib.convlstm import LSTMClassifier, F1, Recall, Precision, Positives, Negatives

class ModelLoader():

    model_loaded = {}
    y_scaler_loaded = False
    x_scaler_loaded = False
    predictors = {}

    def __init__(self, io):
        self.io = io

    def load_scalers(self, save_path):
        """
        Load Scalers
        """
        # Single model
        try:
            fname=save_path+'/xscaler.pkl'
            xscaler = self.io.load_scikit_model(fname)
        except:
            xscaler = None

        try:
            fname=save_path+'/yscaler.pkl'
            yscaler = self.io.load_scikit_model(fname)
        except:
            yscaler = None

        # Dual model
        try:
            fname=save_path+'/xscaler_classifier.pkl'
            xscaler_classifier = self.io.load_scikit_model(fname)
        except:
            xscaler_classifier = None

        try:
            fname=save_path+'/xscaler_regressor.pkl'
            xscaler_regressor = self.io.load_scikit_model(fname)
            fname=save_path+'/yscaler_regressor.pkl'
            yscaler_regressor = self.io.load_scikit_model(fname)
        except:
            xscaler_regressor = None
            yscaler_regressor = None

        return xscaler, yscaler, xscaler_classifier, xscaler_regressor, yscaler_regressor

    def load_keras_model(self, save_path, save_file, force=False, name='predictor'):
        """
        Load tf model
        """
        if not name in self.model_loaded or not self.model_loaded[name] or force:
            logging.info('Loading model from {}'.format(save_path))

            self.io._download_dir_from_bucket(save_path, save_path, force=force)
            dependencies = {
            'F1': F1,
            'Recall': Recall,
            'Precision': Precision,
            'Positives': Positives,
            'Negatives': Negatives
            }
            self.predictors[name] = load_model(save_file, custom_objects=dependencies)
            self.model_loaded[name] = True

        return self.predictors[name]

    def load_classifier(self, save_path, save_file, options):
        """
        Initialize model class (scikit API with a twist) and keras model
        """
        if options.classifier == 'lstm':
            model = self.load_keras_model(save_path, save_file)
            num_of_features = len(options.feature_params)
            if options.month: num_of_features += 1
            params = {'model': model, 'length': options.time_steps, 'batch_size': options.batch_size, 'epochs': options.epochs, 'num_of_features': num_of_features, 'log_dir': options.log_dir}
            classifier = LSTMClassifier(**params)
        else:
            pass #TODO

        return classifier

    def load_tf_model(self, save_path, save_file, name='predictor'):
        """
        Load tf model
        """
        if not self.model_loaded[name]:
            logging.info('Loading model from {}/{}'.format(save_path, save_file))

            self.io._download_dir_from_bucket(save_path, save_path, force=True)
            self.sess = tf.Session()
            saver = tf.train.import_meta_graph(save_file+'.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(save_path))
            graph = tf.get_default_graph()

            # for i in tf.get_default_graph().get_operations():
            #     print(i.name)
            # sys.exit()
            self.X = graph.get_tensor_by_name("inputs/X:0")
            self.op_y_pred = graph.get_tensor_by_name("out_hidden/y_pred/y_pred:0")
            self.model_loaded[name] = True

        return self.sess, self.op_y_pred, self.X

    def load_scikit_model(self, save_file, force=False, name='predictor'):
        """
        Load scikit model
        """
        if not name in self.model_loaded or not self.model_loaded[name] or force:
            logging.info('Loading model from {}'.format(save_file))
            #self.io._download_from_bucket(save_file, save_file)
            self.predictors[name] = self.io.load_scikit_model(save_file)
            self.model_loaded[name] = True

        return self.predictors[name]
