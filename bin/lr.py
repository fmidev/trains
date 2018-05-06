import sys, os
import argparse
import logging
import datetime as dt
import json
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.estimator.export import export

from sklearn.model_selection import train_test_split

from mlfdb import mlfdb
from ml_feature_db.api.mlfdb import mlfdb as db
from lib import io as _io
from lib import viz as _viz

# def serving_input_receiver_fn():
#     serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensors')
#     receiver_tensors      = {"X": serialized_tf_example}
#     feature_spec          = {"X": tf.FixedLenFeature([29],tf.float32)}
#     features              = tf.parse_example(serialized_tf_example, feature_spec)
#     return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def serving_input_receiver_fn():
    """Build the serving inputs."""
    
    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    inputs = {'X': tf.placeholder(dtype=tf.float64)}
    
    #inputs = {}
    #for feat in get_input_columns():
    #    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)    


def lr(features, labels, mode, params):
    """
    model function for linear regression
    """
    # Define parameters
    #batch_size, n_dim = features.shape
    
    # Define placeholders for input
#    X = tf.placeholder(tf.float32, name='X')
#    y = tf.placeholder(tf.float32, name='y')
    if type(features) is dict:
        X = features['X']
    else:
        X = features
    y = labels

    train_losses, val_losses = [], []
            
    W = tf.get_variable("weights", (params['n_dim'], 1),
                        initializer=tf.random_normal_initializer(),
                        dtype=tf.float64)
    b = tf.get_variable("bias", (1, ),
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float64)
    y_pred = tf.matmul(X, W) + b

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput(
                {"pred_output": y_pred}
            )}
        predictions_dict = {"late_minutes": y_pred}
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"y_pred": y_pred},
            export_outputs=export_outputs            
        )
    
    # Define optimizer operation
    loss = tf.reduce_sum((y - y_pred) ** 2 / params['n_samples'])
    optimizer = tf.train.AdamOptimizer()

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimize = optimizer.minimize(
            loss,
            global_step=tf.train.get_global_step()
        )
        
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimize
        )
    
    assert mode == tf.estimator.ModeKeys.EVAL
    rmse = tf.metrics.root_mean_squared_error(labels, y_pred)
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops={'rmse': rmse }
    )

def main():
    """
    Get data from db and save it as csv
    """

    #a = mlfdb.mlfdb()
    a = db.mlfdb()
    io = _io.IO()
    viz = _viz.Viz()

    if not os.path.exists(options.save_path):
        os.makedirs(options.save_path)

    if not os.path.exists(options.output_path):
        os.makedirs(options.output_path)

    starttime, endtime = io.get_dates(options)
        
    logging.info('Loading classification dataset from db')
    if starttime is not None and endtime is not None:
        logging.info('Using time range {} - {}'.format(starttime.strftime('%Y-%m-%d'), endtime.strftime('%Y-%m-%d')))        

    l_metadata, l_header, l_data = a.get_rows(options.dataset,
                                              starttime=starttime,
                                              endtime=endtime,
                                              rowtype='label')

    f_metadata, f_header, f_data = a.get_rows(options.dataset,
                                              starttime=starttime,
                                              endtime=endtime,
                                              rowtype='feature',
                                              parameters=[])
    
    l_metadata, l_data = io.filter_train_type(l_metadata, l_data, traintypes=[0,1], sum_types=True)
    l_metadata, l_data = io.filter_labels(l_metadata, l_data, f_metadata, f_data) #, invert=True)
    
    logging.debug('Labels metadata shape: {} | Labels shape: {}'.format(l_metadata.shape, l_data.shape))
    logging.debug('Features metadata shape: {} | Features shape: {}'.format(f_metadata.shape, f_data.shape))

    target = l_data[:,0]
    X_train, X_test, y_train, y_test = train_test_split(f_data, l_data[:,0], test_size=0.33)

    # Define number of gradient descent loops
    n_loops = 1000
    batch_size = 100
    n_samples, n_dim = X_train.shape

    model = tf.estimator.Estimator(
        model_fn=lr,
        params={
            'n_samples': n_samples,
            'n_dim' : n_dim
        },
        model_dir=options.log_dir
    )

    #    saver = tf.train.Saver()

    model_filename = options.save_path+'/model_state.ckpt'
    export_dir = options.save_path+'/serving'
            
    # Select random mini-batch
    def input_train():
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_train[indices], y_train[indices]
        return X_batch, y_batch

    def input_test():
        indices = np.random.choice(len(X_test), batch_size)
        X_batch, y_batch = X_test[indices], y_test[indices]
        return X_batch, y_batch        
    
    model.train(input_fn=input_train, steps=n_loops)
    model.evaluate(input_fn=input_test, steps=1)

    #feature_spec = {"X": tf.FixedLenFeature([29],tf.float32)}
    #feature_spec = {"X": tf.VarLenFeature(tf.float32)}
    #feature_spec = tf.feature_column.make_parse_example_spec(get_input_columns())
    #serving_input_fn = export.build_parsing_serving_input_receiver_fn(feature_spec)

    model.export_savedmodel(
        export_dir,
        serving_input_receiver_fn
    )
#    io.export_tf_model(sess, export_dir, inputs={'X': X}, outputs={'y': y_pred}, serving_input_fn=serving_input_receiver_fn)

    # filename = options.output_path + '/training_loss.png'
    # viz.plot_learning(np.array(train_losses), np.array(val_losses), filename)
        
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, help='End time of the classification data interval')
    parser.add_argument('--save_path', type=str, default=None, help='Model save path and filename')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--log_dir', type=str, default='/tmp/lr', help='Dataset name')
    parser.add_argument('--db_config_file',
                        type=str,
                        default=None,
                        help='GS address for db config file (if none, ~/.mlfdbconfig is used)')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')
    parser.add_argument('--output_path', type=str, default=None, help='Path where visualizations are saved')
    
    options = parser.parse_args()
    
    if options.save_path is None:
        options.save_path = 'models/'+options.dataset

    if options.output_path is None:
        options.output_path = 'results/'+options.dataset

    debug=False

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
