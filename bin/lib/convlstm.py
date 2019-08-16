import numpy, logging
from keras import backend as K
from keras import losses
from keras.models import Sequential,Model
from keras.layers import Input, Dense, LSTM
from keras import regularizers
from keras import initializers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#from keras.layers import Input

#from keras.layers.convolutional import Conv1D, Conv2D

#from keras.layers.convolutional_recurrent import ConvLSTM2D

#from keras.layers.convolutional import MaxPooling1D

# from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence

# load the dataset but only keep the top n words, zero the rest

# truncate and pad input sequences
#max_review_length = 500
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model

def Recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


class Regression:
    def __init__(self, options, dim):
        """
        Model creation

        options : Object
                  Object with time_steps attribute
        dim : int
              data dimension
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        initializer = initializers.glorot_normal()

        self.model = Sequential()
        self.model.add(LSTM(options.time_steps, input_shape=(options.time_steps, dim)))
        #self.model.add(Dense(12, kernel_initializer=initializer))
        self.model.add(Dense(1, kernel_initializer=initializer))
        self.model.compile(loss=losses.mean_squared_error, optimizer='adam', metrics=[losses.mean_squared_error, losses.mean_absolute_error])

    def get_model(self):
        return self.model


class Classifier:

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        self.model = Sequential()
        self.model.add(LSTM(24, input_shape=(24,19)))
        self.model.add(Dense(4, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', F1, Recall, Precision])

    def get_model(self):
        return self.model


class Autoencoder:

    def __init__(self, input_dim):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        print(input_dim)
        #nb_epoch = 100
        #batch_size = 128
        #input_dim = train_x.shape[1] #num of columns, 30
        encoding_dim = 14
        hidden_dim = int(encoding_dim / 2) #i.e. 7
        learning_rate = 1e-7
        print(hidden_dim)
        input_layer = Input(shape=(input_dim, ), name='input')

        encoder = Dense(encoding_dim, activation="tanh",
                        activity_regularizer=regularizers.l1(learning_rate),
                        name='enc_dense_1')(input_layer)

        encoder = Dense(hidden_dim, activation="relu", name='enc_dense_2')(encoder)

        decoder = Dense(hidden_dim, activation='tanh', name='dec_dense_1')(encoder)
        decoder = Dense(input_dim, activation='relu', name='dec_dense_2')(decoder)

        self.model = Model(inputs=input_layer, outputs=decoder)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', F1, Recall, Precision])

    def get_model(self):
        return self.model
