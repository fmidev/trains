import tensorflow as tf

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
