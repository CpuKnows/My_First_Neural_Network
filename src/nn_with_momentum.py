"""
file:   nn.py
author: John Maxwell
tldr:   Network for forward and backward propagation
"""

from __future__ import print_function
import time
import numpy as np

from units import sigmoid_activation, tanh_activation


class Network(object):

    def __init__(self, layers):
        """
        :param layers: an array of layer sizes
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights with mean 0 and range [-1, 1]
        :return: weights
        """
        weights = [2 * np.random.random([x, x_next]) - 1
                   for x, x_next in zip(self.layers[:-1], self.layers[1:])]
        return weights

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
                layers = self.forward_prop(x_batch)

                # back propagation
                layer_errors, layer_deltas = self.backward_prop(layers, y_batch)

                # update weights
                if momentum:
                    velocities = [velocity_decay * v - learning_rate * np.dot(l.T, d)
                                  for v, l, d in zip(velocities, layers[:-1], layer_deltas)]

                    self.weights = [w + v for w, v in zip(self.weights, velocities)]
                else:
                    self.weights = [w + (-1 * learning_rate * np.dot(l.T, d))
                                    for w, l, d in zip(self.weights, layers[:-1], layer_deltas)]

            if print_error and epoch % 1000 == 0:
                print('Epoch: %.0f \tError: %.4f \telapsed: %.2f ms' %
                      (epoch,
                       np.mean(np.abs(layer_errors[-1])),
                       (time.clock() - start_time) * 1000))

        if print_error:
            print('Training time: %.0f sec' % (time.clock() - start_time_training))

    def forward_prop(self, x):
        layers = list()
        layers.append(np.array([x]).T)

        for w in self.weights[:-1]:
            layers.append(tanh_activation(np.dot(layers[-1], w)))

        layers.append(np.dot(layers[-1], self.weights[-1]))

        return layers

    def backward_prop(self, layers, y):
        layer_errors = [layers[-1] - np.array([y]).T]
        layer_deltas = [layer_errors[-1]]

        for layer_idx in xrange(self.num_layers - 2, 0, -1):
            layer_errors.append(np.dot(layer_deltas[-1], self.weights[layer_idx - 3].T))
            layer_deltas.append(layer_errors[-1] * tanh_activation(layers[layer_idx], True))

        layer_errors.reverse()
        layer_deltas.reverse()

        return layer_errors, layer_deltas
