import numpy as np
import logging
from sklearn.base import BaseEstimator
from math import ceil

from keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from  tensorflow.compat.v1.metrics import percentage_below

import tensorflow as tf
#from tensorflow.keras.backend.tensorflow_backend import set_session
from sklearn.exceptions import NotFittedError

from lib.delaygen import DelayGen


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

def Negatives(y_true, y_pred):
    """Negatives metric.

    Show share of predicted negatives
    """
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    predicted_negatives = tf.cast(K.shape(y_pred)[0], tf.float32) - predicted_positives
    return predicted_negatives / (predicted_positives + predicted_negatives)


def Positives(y_true, y_pred):
    """Positives metric.

    Show share of predicted positives
    """
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    predicted_negatives = tf.cast(K.shape(y_pred)[0], tf.float32) - predicted_positives
    return predicted_positives / (predicted_positives + predicted_negatives)

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

def F1_loss(y_true, y_pred):

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

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
        self.model.add(LSTM(options.time_steps,
                            input_shape=(options.time_steps, dim),
                            activation='sigmoid'),
                            kernel_regularizer=regularizers.l2(0.),
                            dropout=0.3)

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


class LSTMClassifier(BaseEstimator):

    fitted = False

    def __init__(self, length=120, batch_size=1024, epochs=1, num_of_features=19, p_drop=0.3, log_dir=None, class_weight=None, model=None):
        """ Init """
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #set_session(tf.Session(config=config))

        self.length = length
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_dir = log_dir
        self.p_drop = p_drop
        self.class_weight = class_weight

        if model is None:
            self.model = Sequential()
            # in case of other LSTM layer
            #self.model.add(LSTM(length, input_shape=(self.length, num_of_features), activation='tanh', return_sequences=True))
            self.model.add(LSTM(length, input_shape=(self.length, num_of_features), activation='tanh', return_sequences=False))
            self.model.add(Dropout(self.p_drop))
            #self.model.add(LSTM(length, activation='tanh', return_sequences=False))
            #self.model.add(Dropout(self.p_drop))
            self.model.add(Dense(1, activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', F1, Recall, Precision, Negatives, Positives])
            #self.model.compile(loss=F1_loss, optimizer='adam', metrics=['accuracy', F1, Recall, Precision, Negatives, Positives])
            
            #self.model.run_eagerly = True
        else:
            self.fitted = True
            self.model = model

    def save(self, model_filename):
        """ Save """
        self.model.save(model_filename)

    def get_model(self):
        return self.model

    def fit(self, X, y, val_X, val_y):
        """ Fit """
        # data_gen = TimeseriesGenerator(X,
        #                                y,
        #                                length=2, #self.length,
        #                                sampling_rate=1,
        #                                batch_size=10) #self.batch_size)

        data_gen = DelayGen(X, y,
                           length=self.length,
                           sampling_rate=1,
                           batch_size=self.batch_size)

        # data_val_gen = TimeseriesGenerator(val_X,
        #                                    val_y,
        #                                    length=self.length,
        #                                    sampling_rate=1,
        #                                    batch_size=self.batch_size)

        data_val_gen = DelayGen(val_X,
                                val_y,
                                length=self.length,
                                sampling_rate=1,
                                batch_size=self.batch_size)

        patience = 500

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

        d = self.log_dir+'/lstm'
        logging.info('Putting tensorboard logs to {}'.format(d))
        boardcb = TensorBoard(log_dir=d,
                              histogram_freq=5,
                              write_graph=True,
                              profile_batch=0,
                              write_images=True)

        history = self.model.fit(x=data_gen,
                                 validation_data = data_val_gen,
                                 epochs=self.epochs,
                                 shuffle=False,
                                 class_weight=self.class_weight,
                                 callbacks=[es, boardcb]) #, batch_size=64 boardcb,

        self.fitted = True

        return history

    def predict(self, X, type='int'):
        """ Predict """
        if not self.fitted:
            raise NotFittedError()

        self.predict_proba(X)

        # data_gen = TimeseriesGenerator(X,
        #                                np.empty(len(X)),
        #                                length=self.length,
        #                                sampling_rate=1,
        #                                batch_size=self.batch_size)
        #
        #
        # y_pred = self.model.predict_generator(data_gen)
        #y_pred = y_pred[~np.isnan(y_pred)]
        #print(len(y_pred[(y_pred >=.5)]))

        if type == 'bool':
            return np.fromiter(map(lambda x: False if x < .5 else True, self.y_pred), dtype=np.bool)

        # Scale to [-1 1]
        return np.fromiter(map(lambda x: -1 if x < .5 else 1, self.y_pred), dtype=np.int)


    def predict_proba(self, X):
        """ Predict proba """
        if not self.fitted:
            raise NotFittedError()

        data_gen = TimeseriesGenerator(X,
                                       np.empty(len(X)),
                                       length=self.length,
                                       sampling_rate=1,
                                       batch_size=self.batch_size)
        y_pred = self.model.predict_generator(data_gen)

        # Scale to [-1 1]
        y_pred_proba = np.zeros((len(y_pred),2))
        y_pred_proba[:,1] = y_pred.ravel()
        y_pred_proba[:,0] = 1-y_pred.ravel()

        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        return y_pred_proba

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
