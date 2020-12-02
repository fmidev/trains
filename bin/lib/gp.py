import sys, os, logging
import gpflow
from sklearn.base import BaseEstimator
import tensorflow as tf
import numpy as np
from gpflow.utilities import print_summary, positive
from gpflow.utilities.ops import broadcasting_elementwise
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable

class GP(BaseEstimator):
    """
    Scikit API wrapper for gpflow model
    """
    estimator_params = ("Z", "kern", "minibatch_size", "whiten", "length_scale", "likelihood")
    def __init__(self, dim, length_scale=1, kern=None, likelihood=None, Z=None, minibatch_size=1000, whiten=False, n_iter=10000, save_path='/tmp/gp', save_interval=100, output_path='/tmp/gp/output', limit=.5, learning_rate=0.001):
        """
        Initialization
        """
        logging.info('Using GPflow Classifier')
        def invlink(f):
            return gpflow.likelihoods.Bernoulli().invlink(f)

        self.learning_rate = learning_rate
        self.dim = dim
        self.length_scale = length_scale

        self.kern = kern
        if kern is None:
            self.kern = gpflow.kernels.Linear(active_dims=range(dim)) * gpflow.kernels.RationalQuadratic(active_dims=range(dim)) #+ gpflow.kernels.White(active_dims=range(dim))
            #self.kern = Tanimoto()

        self.likelihood = likelihood
        if likelihood is None:
            self.invlink = invlink
            self.likelihood = gpflow.likelihoods.Bernoulli(invlink=self.invlink)
            #self.likelihood = gpflow.likelihoods.Gaussian()
        self.Z = Z
        if Z is None:
            self.Z = np.random.rand(1000, self.dim)
        self.minibatch_size = minibatch_size
        self.whiten = whiten
        self.n_iter = n_iter

        self.limit = limit

        self.output_path = output_path
        self.save_path = save_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.save_interval = save_interval

        self.model = gpflow.models.SVGP(kernel=self.kern,
                                        likelihood=self.likelihood,
                                        inducing_variable=self.Z,
                                        whiten=self.whiten)

    def fit(self, X, y):

        # We turn off training for inducing point locations
        # gpflow.set_trainable(self.model.inducing_variable, False)

        data = (tf.reshape(tf.cast(X, tf.float64), [-1, self.dim]), tf.reshape(tf.cast(y, tf.float64), [-1, 1]))
        print_summary(self.model)

        train_dataset = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(X.shape[0])
        self.logf = self.run_adam(train_dataset)

        return self

    def predict_f(self, X):
        return self.predict_f(X)

    def predict_proba(self, X):
        self.predict(X)
        return self.y_pred_proba

    def predict(self, X, type='int'):
        """
        Predict
        """
        post_mean, self.var, i  = [], [], 0
        i_end = min(i + self.minibatch_size, len(X))
        while i < len(X):
            X_batch = X[i:i_end]
            post_mean_batch, var_batch = self.model.predict_f(tf.cast(X_batch, tf.float64))
            post_mean_batch = self.invlink(post_mean_batch)

            post_mean += post_mean_batch.numpy().ravel().tolist()
            self.var += var_batch.numpy().ravel().tolist()
            i = i_end
            i_end = min(i + self.minibatch_size, len(X))

        post_mean = np.array(post_mean)

        self.y_pred_proba = np.stack([(1-post_mean), post_mean.ravel()], axis=1)

        if type == 'bool':
            return np.fromiter(map(lambda x: False if x < self.limit else True, post_mean), dtype=np.bool)

        # Scale to [0 1]
        return np.fromiter(map(lambda x: 0 if x < self.limit else 1, post_mean), dtype=np.int)


    def save(self, fname, io=None):
        """ Save model as a checkpoint """
        ckpt_path = self.manager.save()
        logging.info(f'Saved to {ckpt_path}')

        print_summary(self.model)

        if io is not None:
            io._upload_dir_to_bucket(self.save_path, self.save_path, ['ckpt', 'checkpoint'])

    def load(self, io=None):
        """ Load model from the checkpoint """
        if io is not None:
            io._download_dir_from_bucket(self.save_path, self.save_path, True)

        step_var = tf.Variable(1, dtype=tf.int32, trainable=False)
        epoch_var = tf.Variable(1, dtype=tf.int32, trainable=False)
        ckpt = tf.train.Checkpoint(model=self.model, step=step_var, epoch=epoch_var)
        ckpt.restore(tf.train.latest_checkpoint(self.save_path))
        logging.info(f"Restored model from {tf.train.latest_checkpoint(self.save_path)} [step:{int(step_var)}, epoch:{int(epoch_var)}]")
        print_summary(self.model)

    def run_adam(self, train_dataset, data=None):
        """
        Utility function running the Adam optimizer
        """
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(train_dataset.batch(self.minibatch_size))
        training_loss = self.model.training_loss_closure(train_iter, compile=True)

        optimizer = tf.optimizers.Adam(learning_rate = self.learning_rate)

        step_var = tf.Variable(1, dtype=tf.int32, trainable=False)
        epoch_var = tf.Variable(1, dtype=tf.int32, trainable=False)
        ckpt = tf.train.Checkpoint(model=self.model, step=step_var, epoch=epoch_var)
        self.manager = tf.train.CheckpointManager(ckpt, self.save_path, max_to_keep=2)

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, self.model.trainable_variables)

        for step in range(self.n_iter):
            optimization_step()

            step_id = step + 1
            if step_id % 10 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)

            if step_id % self.save_interval == 0:
                step_var = step_id
                ckpt_path = self.manager.save()
                elbo = -training_loss().numpy()
                tf.print(f"Epoch {step_id}: ELBO (train) {elbo}, saved at {ckpt_path}")

        return logf





class Tanimoto(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))

        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        # tf.print('Calculating K...')
        if X2 is None:
            #tf.print('X2 none')
            X2 = X

        # tf.print('X')
        # tf.print(X.shape)
        #
        # tf.print('X2')
        # tf.print(X2.shape)


        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        K = self.variance * outer_product/denominator

        # tf.print('K')
        # tf.print(K.shape)
        return K

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        K_diag = tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

        # tf.print('K_diag')
        # tf.print(K_diag.shape)

        return K_diag




class Brownian(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__(active_dims=[0])
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance * tf.minimum(X, tf.transpose(X2))  # this returns a 2D tensor

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

        return self.variance * tf.reshape(X, (-1,))  # this returns a 1D tensor
