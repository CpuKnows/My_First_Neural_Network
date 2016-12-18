"""
file:   nn_with_adv_layers.py
author: John Maxwell
tldr:   Network for forward and backward propagation using layer classes
"""

from __future__ import print_function
import time
import numpy as np

from layer import Layer


class Network(object):

    def __init__(self):
        self.layers = list()
        self.num_layers = 0

    def add_layer(self, layer_type, layer_size, activation=None, activation_prime=None):
        self.layers.append(Layer(layer_type, layer_size, activation, activation_prime))
        self.num_layers += 1

    def clear_layers(self):
        self.layers = list()
        self.num_layers = 0

    def initialize_weights(self):
        """
        Initialize weights with mean 0 and range [-1, 1]

        :return: none
        """

        for l, l_next in zip(self.layers[:-1], self.layers[1:]):
            l.weights = 2 * np.random.random([l.layer_size, l_next.layer_size]) - 1

    def iter_batches(self, x, y, batch_size, shuffle=False):
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

    def mini_batch_learning(self, x, y, n_epoch=10000, batch_size=100, learning_rate=0.01,
                            momentum=False, print_error=False):
        velocity_decay = 0.5
        velocities = [0] * (self.num_layers - 1)

        start_time_training = time.clock()

        for epoch in xrange(n_epoch):
            start_time = time.clock()

            for batch in self.iter_batches(x, y, batch_size, shuffle=True):
                x_batch, y_batch = batch

                # forward propagation
                forward_out = np.array([x_batch]).T
                for l in self.layers:
                    forward_out = l.forward_prop(forward_out)

                # back propagation
                layer_errors, layer_deltas = list(), list()

                for l in reversed(self.layers[1:]):
                    if len(layer_deltas) == 0:
                        l_error, l_delta = l.backward_prop(np.array([y_batch]).T)
                    else:
                        l_error, l_delta = l.backward_prop(layer_deltas[-1])
                    layer_errors.append(l_error)
                    layer_deltas.append(l_delta)

                layer_errors.reverse()
                layer_deltas.reverse()

                # update weights
                if momentum:
                    velocities = [velocity_decay * v - (learning_rate / batch_size) * np.dot(l.node_values.T, d)
                                  for v, l, d in zip(velocities, self.layers[:-1], layer_deltas)]

                    for l, v in zip(self.layers[:-1], velocities):
                        l.weights += v

                else:
                    for l, d in zip(self.layers[:-1], layer_deltas):
                        l.weights += -1 * learning_rate * np.dot(l.node_values.T, d)

            if print_error and epoch % 1000 == 0:
                print('Epoch: %.0f \tError: %.4f \telapsed: %.2f ms' %
                      (epoch,
                       np.mean(np.abs(layer_errors[-1])),
                       (time.clock() - start_time) * 1000))

        if print_error:
            print('Training time: %.0f sec' % (time.clock() - start_time_training))

    def forward_prop(self, x):
        forward_out = x
        for l in self.layers:
            forward_out = l.forward_prop(forward_out)

        return forward_out
