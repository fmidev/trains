import logging
import tensorflow as tf
from tensorflow.python.framework import constant_op

def model_func(features, labels, mode, params):
    """
    model function for linear regression
    """
    # Define parameters

    # Define placeholders for input
#    X = tf.placeholder(tf.float32, name='X')
#    y = tf.placeholder(tf.float32, name='y')
    if type(features) is dict:
        X = features['X']
    else:
        X = features
    y = labels

    try:
        n_samples, n_dim = X.shape
    except ValueError:
        n_samples = None
        n_dim = params['n_dim']

    logging.debug('n_dim: {} | n_smaples: {}'.format(n_dim, n_samples))
    train_losses, val_losses = [], []


    W = tf.get_variable("weights", (n_dim, 1),
                        initializer=tf.random_normal_initializer(),
                        dtype=tf.float32)
    b = tf.get_variable("bias", (1, ),
                        initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
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
    reg = tf.nn.l2_loss(W)
    loss = tf.reduce_mean(tf.square(y - y_pred)) + reg
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

    # Metrics
    rmse = tf.metrics.root_mean_squared_error(labels, y_pred)

    def r_squared(labels, y_pred):
        unexplained_error = tf.reduce_sum(tf.square((labels - y_pred)))
        total_error = tf.reduce_sum(tf.square((labels - tf.reduce_mean(labels))))
        r2 = tf.subtract(tf.constant(1., dtype='float32'), tf.div(unexplained_error, total_error))
        return r2, constant_op.constant(1.)

    metrics = {'rmse': rmse,
               'mae': tf.metrics.mean_absolute_error(labels, y_pred),
               'rmse_below_10': tf.metrics.percentage_below(rmse, 10),
               'rmse_below_5': tf.metrics.percentage_below(rmse, 5),
               'rmse_below_3': tf.metrics.percentage_below(rmse, 3),
               'rmse_below_1': tf.metrics.percentage_below(rmse, 1),
               'y_pred_below_10': tf.metrics.percentage_below(y_pred, 10),
               'y_pred_below_5': tf.metrics.percentage_below(y_pred, 5),
               'y_pred_below_3': tf.metrics.percentage_below(y_pred, 3),
               'y_pred_below_1': tf.metrics.percentage_below(y_pred, 1),
               'r2': r_squared(labels, y_pred)
               }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=metrics
    )
