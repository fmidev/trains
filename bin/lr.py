import sys, os
import argparse
import logging
import datetime as dt
import json
import itertools
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from mlfdb import mlfdb
from ml_feature_db.api.mlfdb import mlfdb as db
from lib import io as _io
from lib import viz as _viz


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
    
    # Define parameters
    n_samples, n_dim = X_train.shape
    batch_size = 100
    
    # Define placeholders for input
    X = tf.placeholder(tf.float32, name='X')
    y = tf.placeholder(tf.float32, name='y')

    train_losses, val_losses = [], []
    
    # Define variables to be learned
    with tf.variable_scope("linear-regression"):
        
        W = tf.get_variable("weights", (n_dim, 1),
                            initializer=tf.random_normal_initializer())
        b = tf.get_variable("bias", (1, ),
                            initializer=tf.constant_initializer(0.0))
        y_pred = tf.matmul(X, W) + b
        loss = tf.reduce_sum((y - y_pred) ** 2 / n_samples)

        # Define optimizer operation
        optimizer = tf.train.AdamOptimizer()
        optimize = optimizer.minimize(loss)

        # Define number of gradient descent loops
        n_loops = 1000

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        model_filename = options.save_path+'/model_state.ckpt'
        export_dir = options.save_path+'/serving'


        def serving_input_receiver_fn():
            """Build the serving inputs."""

            # The outer dimension (None) allows us to batch up inputs for
            # efficiency. However, it also means that if we want a prediction
            # for a single instance, we'll need to wrap it in an outer list.
            inputs = {"X": tf.placeholder(shape=[None, n_dim], dtype=tf.float32)}
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        
        with tf.Session() as sess:
            # Initialize Variables in graph
            sess.run(init_op)

            for step_idx in range(n_loops):
                # Select random mini-batch
                indices = np.random.choice(n_samples, batch_size)
                X_batch, y_batch = X_train[indices], y_train[indices]

                # Perform a single gradient descent step
                _, loss_val = sess.run([optimize, loss],
                                       feed_dict={X: X_batch, y: y_batch})
                val_loss_val = sess.run([loss],
                                        feed_dict={X: X_test, y: y_test})
                
                train_losses.append(loss_val)
                val_losses.append(val_loss_val)
                
                # Print training status
                if step_idx % 100 == 0:
                    saver.save(sess, model_filename)
                    print("[{}] train loss: {}".format(step_idx, loss_val))
                    print("[{}] test loss: {}".format(step_idx, val_loss_val))

            W_val, b_val = sess.run([W, b])
            io.export_tf_model(sess, export_dir, inputs={'X': X}, outputs={'y': y_pred})

    filename = options.output_path + '/training_loss.png'
    viz.plot_learning(np.array(train_losses), np.array(val_losses), filename)
        
    
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, help='End time of the classification data interval')
    parser.add_argument('--save_path', type=str, default=None, help='Model save path and filename')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
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
