"""
file:   sgd.py
author: John Maxwell
tldr:   Network for stochastic gradient descent
"""

from __future__ import print_function
import time
import numpy as np

from blob import Blob
from layer import Layer
from error_functions import accuracy, logloss


class SGD(object):
    def __init__(self):
        """
        Initialize network layers
        """
        self.layers = list()
        self.num_layers = 0
        self.blobs = list()
        self.num_blobs = 0
        self.training_error = list()
        self.validation_error = list()

    def add_layer(self, new_layer):
        """
        Adds a layer to the network

        :param new_layer: layer to add
        :return: none
        """
        isinstance(new_layer, Layer)

        self.layers.append(new_layer)
        self.num_layers += 1

    def clear_layers(self):
        """
        remove all layers

        :return: none
        """
        self.layers = list()
        self.num_layers = 0
        self.blobs = list()
        self.num_blobs = 0

    def initialize_blobs(self):
        """
        Initialize blobs to be the appropriate size.

        :return: none
        """
        self.blobs = list()
        self.num_blobs = 0

        for l, l_next in zip(self.layers[:-1], self.layers[1:]):
            self.blobs.append(Blob(l, l_next))
            self.num_blobs += 1

            l.top = self.blobs[-1]
            l_next.bottom = self.blobs[-1]

        self.blobs.append(Blob(self.layers[-1], None))
        self.num_blobs += 1

        self.layers[-1].top = self.blobs[-1]

    def clear_blobs(self):
        """
        remove all blobs

        :return: none
        """
        self.blobs = list()
        self.num_blobs = 0

        for l in self.layers:
            l.bottom = None
            l.top = None

    def initialize_weights(self):
        """
        Initialize weights with mean 0 and range [-1, 1]

        :return: none
        """
        self.initialize_blobs()

        for b in self.blobs[:-1]:
            b.set_weights(2 * np.random.random(b.shape) - 1)

    def iter_batches(self, x, y, batch_size, shuffle=False):
        """
        Creates mini-batches for SGD.

        :param x: feature data
        :param y: ground truth
        :param batch_size: number of training cases in a mini-batch
        :param shuffle: T/F randomize order
        :return: a mini-batch of x and y
        """
        assert x.shape[0] == y.shape[0]

        if shuffle:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)

        for start_idx in range(0, x.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                batch_indices = indices[start_idx:start_idx + batch_size]
            else:
                batch_indices = slice(start_idx, start_idx + batch_size)

            yield x[batch_indices], y[batch_indices]

    def mini_batch_learning(self, x, y, x_val=None, y_val=None, n_epoch=10000, batch_size=100, learning_rate=0.01,
                            momentum=False, cost_function='accuracy', print_error=False, print_every=None):
        """
        Handles forward and backward propagation for learning.

        :param x: feature data
        :param y: ground truth
        :param x_val: validation feature data
        :param y_val: validation ground truth
        :param n_epoch: number of epochs
        :param batch_size: number of training cases in a mini-batch
        :param learning_rate: learning rate parameter
        :param momentum: T/F use momentum
        :param cost_function: accuracy or logloss
        :param print_error: T/F print error every so often
        :param print_every: how often to print error
        :return: none
        """
        self.training_error = list()
        self.validation_error = list()

        velocity_decay = 0.5
        velocities = [0] * (self.num_layers - 1)

        start_time_training = time.clock()

        for epoch in xrange(n_epoch):
            start_time = time.clock()

            for batch in self.iter_batches(x, y, batch_size, shuffle=True):
                x_batch, y_batch = batch

                # forward propagation
                self.forward_prop(x_batch)

                # back propagation
                layer_errors, layer_deltas = self.backward_prop(y_batch)

                # update weights
                if momentum:
                    velocities = [velocity_decay * v - (learning_rate / batch_size) * np.dot(b.activation_values.T, d)
                                  for v, b, d in zip(velocities, self.blobs[:-1], layer_deltas)]

                    for b, v in zip(self.blobs[:-1], velocities):
                        b.set_weights(b.get_weights() + v)

                else:
                    for b, d in zip(self.blobs[:-1], layer_deltas):
                        b.set_weights(b.get_weights() + -learning_rate / batch_size * np.dot(b.activation_values.T, d))

            if x_val is not None and y_val is not None:
                prediction_train = self.forward_prop(x)
                prediction_val = self.forward_prop(x_val)
                if cost_function is 'accuracy':
                    self.training_error.append(1 - accuracy(prediction_train, y))
                    self.validation_error.append(1 - accuracy(prediction_val, y_val))
                elif cost_function is 'logloss':
                    self.training_error.append(logloss(prediction_train, y))
                    self.validation_error.append(logloss(prediction_val, y_val))

            if (print_error and epoch % print_every == 0) or epoch == n_epoch - 1:
                time_passed = time.clock() - start_time

                prediction_train = self.forward_prop(x)
                if x_val is not None and y_val is not None:
                    prediction_val = self.forward_prop(x_val)

                    if cost_function is 'accuracy':
                        print('Epoch: %.0f \tTrain Error: %.4f \tValidation Error: %.4f \tElapsed: %.0f s' %
                              (epoch,
                               1 - accuracy(prediction_train, y),
                               1 - accuracy(prediction_val, y_val),
                               time_passed))
                    elif cost_function is 'logloss':
                        print('Epoch: %.0f \tTrain Error: %.4f \tValidation Error: %.4f \tElapsed: %.0f s' %
                              (epoch,
                               logloss(prediction_train, y),
                               logloss(prediction_val, y_val),
                               time_passed))
                else:
                    if cost_function is 'accuracy':
                        print('Epoch: %.0f \tTrain Error: %.4f \tElapsed: %.0f s' %
                              (epoch,
                               1 - accuracy(prediction_train, y),
                               time_passed))
                    elif cost_function is 'logloss':
                        print('Epoch: %.0f \tTrain Error: %.4f \tElapsed: %.0f s' %
                              (epoch,
                               logloss(prediction_train, y),
                               time_passed))

        if print_error:
            print('Training time: %.0f sec' % (time.clock() - start_time_training))

    def forward_prop(self, x):
        """
        Forward propagation through the network.

        :param x: input feature data
        :return: forward propagation of network
        """
        forward_out = x
        for l in self.layers:
            forward_out = l.forward_prop(forward_out)

        return forward_out

    def backward_prop(self, y):
        """
        Backward propagation through the network

        :param y: target values
        :return: layer_errors, layer_deltas
        """
        layer_errors, layer_deltas = list(), list()

        for l in reversed(self.layers[1:]):
            if len(layer_deltas) == 0:
                l_error, l_delta = l.backward_prop(y)
            else:
                l_error, l_delta = l.backward_prop(layer_deltas[-1])
            layer_errors.append(l_error)
            layer_deltas.append(l_delta)

        layer_errors.reverse()
        layer_deltas.reverse()
        return layer_errors, layer_deltas

    def vectorized_result(self, j):
        """
        Return a 10-dimensional unit vector with a 1.0 in the j'th position
        and zeroes elsewhere.  This is used to convert a digit (0...9)
        into a corresponding desired output from the neural network.
        """
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e
