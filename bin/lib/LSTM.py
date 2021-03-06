import tensorflow as tf

class LSTM(object):

    def __init__(self, n_steps, input_size, output_size, n_hidden, lr, p_drop=0, batch_size=None):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.lr = lr
        self.p_drop = p_drop
        self.batch_size = batch_size

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [n_steps, None, input_size], name='X')
            self.y = tf.placeholder(tf.float32, [n_steps, None], name='y')

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('LSTM_cell'):
            self.add_cell()

        with tf.variable_scope('out_dense_1'):
            #in_size = self.cell_outputs.shape[1]
            in_size = input_size
            out_size = int(self.n_steps/2)
            self.dense_1 = self.add_dense_layer(self.cell_outputs, in_size, out_size)

        # with tf.variable_scope('out_dense_2'):
        #     in_size = out_size
        #     out_size = int(out_size/2)
        #     self.dense_2 = self.add_dense_layer(self.dense_1, in_size, out_size)

        with tf.variable_scope('out_hidden'):
            self.add_output_layer(self.dense_1, out_size)

        with tf.name_scope('loss'):
            self.compute_loss()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def add_input_layer(self):
        l_in_x = tf.reshape(self.X, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # W (in_size, n_hidden)
        W_in = self._weight_variable([self.input_size, self.n_hidden])
        # bs (n_hidden, )
        b_in = self._bias_variable([self.n_hidden,])
        # l_in_y = (batch * n_steps, n_hidden)
        with tf.name_scope('WX_plus_b'):
            l_in_y = tf.matmul(l_in_x, W_in) + b_in
        # reshape l_in_y ==> (batch, n_steps, n_hidden)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.n_hidden], name='2_3D')
        self.l_in_y_drop = tf.nn.dropout(self.l_in_y, self.p_drop)


    def add_cell(self):
        lstm_cell = tf.contrib.rnn.LSTMCell(self.n_hidden, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                         self.l_in_y_drop,
                                                                         dtype=tf.float32,
                                                                         time_major=True)

    def add_dense_layer(self, input, in_size, out_size):

        # self.out_dense_size = int(self.n_steps/2)

        # shape = (batch * steps, n_hidden)
        l_dense_x = tf.reshape(input, [-1, self.n_hidden], name='2_2D')
        #in_size = self.batch_size * self.n_steps
        #l_dense_x = tf.reshape(input, [self.n_hidden, -1], name='2_2D')

        #l_out_x_drop = tf.nn.dropout(l_out_x, self.p_drop)
        W_dense = self._weight_variable([self.n_hidden, out_size])
        b_dense = self._bias_variable([out_size, ])

        return tf.sigmoid(tf.matmul(l_dense_x, W_dense) + b_dense, name='act_dense')


    def add_output_layer(self, input, in_size):

        # shape = (batch * steps, n_hidden)
        l_out_x = tf.reshape(input, [-1, in_size], name='2_2D')
        #l_out_x_drop = tf.nn.dropout(l_out_x, self.p_drop)
        W_out = self._weight_variable([in_size, self.output_size])
        b_out = self._bias_variable([self.output_size, ])

        # shape = (batch * steps, output_size)
        with tf.name_scope('pred'):
            self.pred = tf.matmul(l_out_x, W_out) + b_out
        with tf.name_scope('y_pred'):
            #self.y_pred = tf.cast(tf.rint(tf.reshape(self.pred, [self.n_steps,-1], name='reshaped_pred')), tf.uint8, name='y_pred')
            self.y_pred = tf.reshape(self.pred, [self.n_steps,-1], name='y_pred')


    def compute_loss(self):
        with tf.name_scope('loss'):
            reshape_target = tf.reshape(self.y, [-1], name='reshape_target')
            reshape_pred = tf.reshape(self.pred, [-1], name='reshape_pred')

            # MSE
            self.loss = tf.reduce_mean(tf.squared_difference(reshape_target, reshape_pred))

            # MSE^2
            # self.loss = tf.reduce_mean(tf.square(tf.squared_difference(reshape_target, reshape_pred)))

            # MSE^3
            #self.loss = tf.reduce_mean(tf.pow(tf.squared_difference(reshape_target, reshape_pred), 3))

            # KLD
            #tf.contrib.distributions.
            self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(reshape_target, reshape_pred)))
            self.mse = tf.square(tf.reduce_mean(tf.squared_difference(reshape_target, reshape_pred)))
            self.mae = tf.reduce_mean(tf.losses.absolute_difference(reshape_target, reshape_pred))

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.glorot_uniform_initializer()
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)



class LSTM_class(object):

    def __init__(self, n_steps, input_size, output_size, n_hidden, lr, p_drop=0, batch_size=None):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.lr = lr
        self.p_drop = p_drop
        self.batch_size = batch_size

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, [n_steps, None, input_size], name='X')
            self.y = tf.placeholder(tf.int32, [n_steps, None], name='y')

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('LSTM_cell'):
            self.add_cell()

        with tf.variable_scope('out_dense_1'):
            #in_size = self.cell_outputs.shape[1]
            in_size = input_size
            out_size = int(self.n_steps/2)
            self.dense_1 = self.add_dense_layer(self.cell_outputs, in_size, out_size)

        # with tf.variable_scope('out_dense_2'):
        #     in_size = out_size
        #     out_size = int(out_size/2)
        #     self.dense_2 = self.add_dense_layer(self.dense_1, in_size, out_size)

        with tf.variable_scope('out_hidden'):
            self.add_output_layer(self.dense_1, out_size)

        with tf.name_scope('loss'):
            self.compute_loss()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def add_input_layer(self):
        l_in_x = tf.reshape(self.X, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # W (in_size, n_hidden)
        W_in = self._weight_variable([self.input_size, self.n_hidden])
        # bs (n_hidden, )
        b_in = self._bias_variable([self.n_hidden,])
        # l_in_y = (batch * n_steps, n_hidden)
        with tf.name_scope('WX_plus_b'):
            l_in_y = tf.matmul(l_in_x, W_in) + b_in
        # reshape l_in_y ==> (batch, n_steps, n_hidden)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.n_hidden], name='2_3D')
        self.l_in_y_drop = tf.nn.dropout(self.l_in_y, self.p_drop)


    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                         self.l_in_y_drop,
                                                                         dtype=tf.float32,
                                                                         time_major=True)

    def add_dense_layer(self, input, in_size, out_size):

        # self.out_dense_size = int(self.n_steps/2)

        # shape = (batch * steps, n_hidden)
        l_dense_x = tf.reshape(input, [-1, self.n_hidden], name='2_2D')
        #in_size = self.batch_size * self.n_steps
        #l_dense_x = tf.reshape(input, [self.n_hidden, -1], name='2_2D')

        #l_out_x_drop = tf.nn.dropout(l_out_x, self.p_drop)
        W_dense = self._weight_variable([self.n_hidden, out_size])
        b_dense = self._bias_variable([out_size, ])

        return tf.sigmoid(tf.matmul(l_dense_x, W_dense) + b_dense, name='act_dense')


    def add_output_layer(self, input, in_size):

        # shape = (batch * steps, n_hidden)
        l_out_x = tf.reshape(input, [-1, in_size], name='2_2D')
        l_out_x_drop = tf.nn.dropout(l_out_x, self.p_drop)
        W_out = self._weight_variable([in_size, self.output_size])
        b_out = self._bias_variable([self.output_size, ])

        # shape = (batch * steps, output_size)
        with tf.name_scope('pred_proba'):
            self.pred_proba = tf.matmul(l_out_x, W_out) + b_out

        with tf.name_scope('pred'):
            #self.pred = tf.matmul(l_out_x, W_out) + b_out
            self.pred = tf.nn.softmax(self.pred_proba)
        with tf.name_scope('y_pred'):
            #self.y_pred = tf.cast(tf.rint(tf.reshape(self.pred, [self.n_steps,-1], name='reshaped_pred')), tf.uint8, name='y_pred')
            self.y_pred = tf.reshape(self.pred, [self.n_steps,-1], name='y_pred')


    def compute_loss(self):
        with tf.name_scope('loss'):
            reshape_target = tf.reshape(self.y, [-1], name='reshape_target')
            reshape_pred = tf.reshape(self.y_pred, [-1], name='reshape_pred')

            # TODO add weights?
            self.loss = tf.losses.sparse_softmax_cross_entropy(reshape_target, self.pred_proba)

            self.accuracy =  tf.metrics.accuracy(reshape_target, reshape_pred)
            # TODO F1, precision, recall

            # MSE
            #self.loss = tf.reduce_mean(tf.squared_difference(reshape_target, reshape_pred))

            # MSE^2
            # self.loss = tf.reduce_mean(tf.square(tf.squared_difference(reshape_target, reshape_pred)))

            # MSE^3
            #self.loss = tf.reduce_mean(tf.pow(tf.squared_difference(reshape_target, reshape_pred), 3))

            # KLD
            #tf.contrib.distributions.
            #self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(reshape_target, reshape_pred)))
            #self.mse = tf.square(tf.reduce_mean(tf.squared_difference(reshape_target, reshape_pred)))
            #self.mae = tf.reduce_mean(tf.losses.absolute_difference(reshape_target, reshape_pred))


    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.glorot_uniform_initializer()
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
