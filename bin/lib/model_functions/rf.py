import logging
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils

def serving_input_receiver_fn():
    """Build the serving inputs."""

    # The outer dimension (None) allows us to batch up inputs for
    # efficiency. However, it also means that if we want a prediction
    # for a single instance, we'll need to wrap it in an outer list.
    inputs = {'X': tf.placeholder(dtype=tf.float32, shape=(19, 1))}
    #inputs = {'X': tf.VarLenFeature(dtype=tf.float32)}

    x = tf.feature_column.numeric_column("X")
    feature_columns = [x]
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    #inputs = {}
    #for feat in get_input_columns():
    #    inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
    #return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)
    # print(tf.estimator.export.ServingInputReceiver(inputs, inputs))
    #feature_spec = tf.feature_column.make_parse_example_spec(inputs)

    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    #print(export_input_fn())
    # return export_input_fn()

    # # return tf.contrib.learn.build_parsing_serving_input_fn(feature_spec)
    # serialized_tf_example = tf.placeholder(dtype=tf.string,
    #                                         name='input_example_tensor')
    # receiver_tensors = {'examples': serialized_tf_example}
    # features = tf.parse_example(serialized_tf_example, inputs)
    # return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


    # inputOps = input_fn_utils.InputFnOps(
    #      feature_spec,
    #      None,
    #      feature_spec)
    # print(inputOps)
    # return inputOps
    print(tf.estimator.export.ServingInputReceiver(inputs, inputs))
    features, receiver_tensors, receiver_tensors_alternatives = tf.estimator.export.ServingInputReceiver(inputs, inputs)
    iops = input_fn_utils.InputFnOps(features, receiver_tensors, receiver_tensors_alternatives)
    #print(iops)
    #return iops
    #return {'features': features, 'receiver_tensors': receiver_tensors}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


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
    y_pred = labels

    try:
        n_samples, n_dim = X.shape
    except ValueError:
        n_samples = None
        n_dim = params['n_dim']

    logging.debug('n_dim: {} | n_smaples: {}'.format(n_dim, n_samples))
    train_losses, val_losses = [], []

    num_steps = 2 # Total steps to train
    batch_size = 1024 # The number of samples per batch
    num_classes = 1000
    num_features = params['n_dim']
    num_trees = 10
    max_nodes = 1000

    #X = tf.placeholder(tf.float32, shape=[None, num_features], name='X')
    #y_pred = tf.placeholder(tf.float32, shape=[None], name='y_pred')

    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes).fill()


    forest_graph = tensor_forest.RandomForestGraphs(hparams)

    #y_pred = tf.matmul(X, W) + b

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
    train_op = forest_graph.training_graph(X, y_pred)
    loss = forest_graph.training_loss(X, y_pred)
    infer_op, _, _ = forest_graph.inference_graph(X)

    # optimizer = tf.train.AdagradOptimizer(0.05)
    # opt = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimize = train_op.minimize(
        #    loss
        #)
        print('in mode TRAIN')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=infer_op
        )

    assert mode == tf.estimator.ModeKeys.EVAL

    # Metrics
    rmse = tf.metrics.root_mean_squared_error(labels, y_pred)

    def r_squared(labels, y_pred):
        unexplained_error = tf.reduce_sum(tf.square((labels - y_pred)))
        total_error = tf.reduce_sum(tf.square((labels - tf.reduce_mean(labels))))
        r2 = tf.subtract(tf.constant(1., dtype='float64'), tf.div(unexplained_error, total_error))
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
