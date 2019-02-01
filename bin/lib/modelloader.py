import sys, os, argparse, logging, json

import tensorflow as tf
from tensorflow.python.estimator.export import export
from tensorflow.python.framework import constant_op
from tensorflow.contrib import predictor

from sklearn import metrics

class ModelLoader():

    model_loaded = False

    def __init__(self, io):
        self.io = io

    def load_tf_model(self, save_path, save_file):
        """
        Load tf model
        """
        if not self.model_loaded:
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
            self.model_loaded = True

        return self.sess, self.op_y_pred, self.X

    def load_scikit_model(self, save_path, save_file):
        """
        Load scikit model
        """
        if not self.model_loaded:
            logging.info('Loading model from {}/{}'.format(save_path, save_file))
            self.io._download_from_bucket(save_file, save_file)
            self.predictor = self.io.load_scikit_model(save_file)
            self.model_loaded = True

        return self.predictor
