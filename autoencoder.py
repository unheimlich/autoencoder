import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
import numbers

import theano
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer, dropout, get_output, get_all_params
from lasagne.nonlinearities import rectify, linear
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum

__author__ = 'andrew'


class Autoencoder(object):

    def __init__(self, num_vars=3, num_channels=1, nodes=32, learning_rate=0.001, num_epochs=500,
                 dropout_rate=0.5, batch_size=100,learning_rate_decay=1.0, activation=rectify,
                 validate_pct=0.1, momentum=0.9, verbose=False):

        self.num_vars = num_vars
        self.num_channels = num_channels
        self.nodes = nodes
        self.learning_rate = theano.shared(np.asarray(learning_rate, dtype=theano.config.floatX))
        self.learning_rate_decay = learning_rate_decay
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.validate_pct = validate_pct
        self.network = None
        self.hidden = None
        self.train_fn = None
        self.val_fn = None
        self.input_var = T.matrix('inputs', dtype='float32')
        self.target_var = T.matrix('targets', dtype='float32')
        self.num_vars = num_vars
        self.num_channels = num_channels
        self.decay_learning_rate = theano.function(inputs=[],
                                                   outputs=self.learning_rate,
                                                   updates={self.learning_rate:
                                                            self.learning_rate * learning_rate_decay})

        self.activation = activation
        self.verbose = verbose

        self.build_ae()
        self.build_train_fn()

    def predict(self, x):

        transform = get_output(self.network, deterministic=True)
        transform_fn = theano.function([self.input_var], transform)

        transformed = np.zeros((1, self.num_vars), dtype=np.float32)

        batches = 0
        for batch in self.iterate_minibatches(x, self.batch_size, shuffle=False):
                inputs = batch
                transformed = np.concatenate((transformed, transform_fn(inputs)), 0)
                batches += 1

        return np.array(transformed[1:], dtype=np.float32)

    def transform(self, x):

        projection = get_output(self.hidden, deterministic=True)
        projection_fn = theano.function([self.input_var], projection)

        projected = np.zeros((1, self.nodes))

        batches = 0
        for batch in self.iterate_minibatches(x, self.batch_size, shuffle=False):
                inputs = batch
                projected = np.concatenate((projected, projection_fn(inputs)), 0)
                batches += 1

        return np.array(projected[1:], dtype=np.float32)

    def fit(self, x):

        validate_flag = self.validate_pct > 0.0

        if validate_flag:

            self.build_validate_fn()
            x, x_val = self.train_validate_split(x)

        print("Starting training...")

        for epoch in range(self.num_epochs):

            start_time = time.time()
            self.run_epoch(x)

            if validate_flag:

                self.run_epoch_validate(x_val)

            self.decay_learning_rate()

            if self.verbose:

                print("Epoch {} of {} took {:.3f}s".format(epoch + 1, self.num_epochs, time.time() - start_time))

    def build_ae(self):

        input_layer = InputLayer(shape=(None, self.num_vars*self.num_channels), input_var=self.input_var)

        self.hidden = DenseLayer(dropout(input_layer, p=self.dropout_rate),
                                 num_units=self.nodes,
                                 nonlinearity=self.activation)

        self.network = DenseLayer(self.hidden,
                                  num_units=self.num_vars,
                                  W=self.hidden.W.T,
                                  nonlinearity=linear)

    def build_train_fn(self):

        prediction = get_output(self.network, deterministic=False)

        loss = squared_error(prediction, self.target_var)
        loss = loss.mean()

        params = get_all_params(self.network, trainable=True)

        updates = nesterov_momentum(loss, params, learning_rate=self.learning_rate, momentum=self.momentum)

        self.train_fn = theano.function([self.input_var, self.target_var], loss, updates=updates)

    def build_validate_fn(self):

        prediction = get_output(self.network, deterministic=True)
        loss = squared_error(prediction, self.target_var)
        loss = loss.mean()

        self.val_fn = theano.function([self.input_var, self.target_var], loss)

    def train_validate_split(self, x):

        num_obs = x.shape[0]
        val_size = np.round(num_obs*self.validate_pct)

        indices = np.arange(num_obs)
        np.random.shuffle(indices)

        x = x[indices]

        x, x_val = x[0:num_obs-val_size], x[num_obs-val_size:num_obs]

        return x, x_val

    def run_epoch(self, x):

        train_err = 0
        train_batches = 0
        for batch in self.iterate_minibatches(x, self.batch_size, shuffle=True):
            inputs = batch
            train_err += self.train_fn(inputs, inputs)
            train_batches += 1

        if self.verbose:

            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    def run_epoch_validate(self, x_val):

        val_err = 0
        val_batches = 0
        for batch in self.iterate_minibatches(x_val, self.batch_size, shuffle=False):
            inputs = batch
            err = self.val_fn(inputs, inputs)
            val_err += err
            val_batches += 1

        if self.verbose:

            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    def iterate_minibatches(self, inputs, batch_size, shuffle=False):

        if shuffle:

            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)

        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):

            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            yield inputs[excerpt]


class Autoencoder2D(Autoencoder):

    def __init__(self, nodes=32, learning_rate=0.001, num_epochs=500, dropout_rate=0.5, batch_size=1,
                 learning_rate_decay=1.0, activation=rectify, validate_pct=0.1, momentum=0.9, filter_size=(5, 5),
                 num_vars=25, num_channels=1, verbose=False):

        super(Autoencoder2D, self).__init__(num_vars=num_vars, num_channels=num_channels, nodes=nodes,
                                            learning_rate=learning_rate, num_epochs=num_epochs,
                                            dropout_rate=dropout_rate, batch_size=batch_size,
                                            learning_rate_decay=learning_rate_decay, activation=activation,
                                            validate_pct=validate_pct, momentum=momentum, verbose=verbose)

        self.filter_size = filter_size

    def iterate_minibatches(self, inputs, batch_size, shuffle=False):

        if shuffle:

            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)

        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):

            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            yield self.extract_patches(inputs[excerpt][0])

    def extract_patches(self, arr, extraction_step=1):

        arr_ndim = arr.ndim

        patch_shape = self.filter_size

        if isinstance(extraction_step, numbers.Number):
            extraction_step = tuple([extraction_step] * arr_ndim)

        patch_strides = arr.strides

        slices = [slice(None, None, st) for st in extraction_step]
        indexing_strides = arr[slices].strides

        patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                               np.array(extraction_step)) + 1

        shape = tuple(list(patch_indices_shape) + list(patch_shape))
        strides = tuple(list(indexing_strides) + list(patch_strides))

        patches = as_strided(arr, shape=shape, strides=strides)

        return patches.reshape((np.prod(patch_indices_shape), np.prod(np.array(patch_shape))))
