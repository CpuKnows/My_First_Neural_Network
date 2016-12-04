"""
file:   nn.py
author: John Maxwell
tldr:   Network for forward and backward propagation
"""

from __future__ import print_function
import numpy as np

from units import sigmoid, tanh


class Network(object):

    def __init__(self):
        np.random.seed(42)

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

    def mini_batch_learning(self, x, y, n_epoch=10000, batch_size=100, learning_rate=0.01, print_error=False):

        # Initialize weights with mean 0 and range [-1, 1]
        syn0 = 2 * np.random.random([1, 10]) - 1
        syn1 = 2 * np.random.random([10, 10]) - 1
        syn2 = 2 * np.random.random([10, 1]) - 1

        for epoch in xrange(n_epoch):
            for batch in self.iter_batches(x, y, batch_size, shuffle=True):
                x_batch, y_batch = batch

                # forward propagation
                l0 = np.array([x_batch]).T
                l1 = tanh(np.dot(l0, syn0))
                l2 = tanh(np.dot(l1, syn1))
                l3 = np.dot(l2, syn2)

                # calc error
                l3_error = l3 - np.array([y_batch]).T

                # back propagation
                l3_delta = l3_error

                l2_error = np.dot(l3_delta, syn2.T)
                l2_delta = l2_error * tanh(l2, True)

                l1_error = np.dot(l2_delta, syn1.T)
                l1_delta = l1_error * tanh(l1, True)

                # update weights
                syn2 += -1 * learning_rate * np.dot(l2.T, l3_delta)
                syn1 += -1 * learning_rate * np.dot(l1.T, l2_delta)
                syn0 += -1 * learning_rate * np.dot(l0.T, l1_delta)

            if print_error and epoch % 1000 == 0:
                print('Error:', np.mean(np.abs(l3_error)))

        return syn0, syn1, syn2

    def forward_prop(self, x, weights, y=None):
        syn0, syn1, syn2 = weights

        # forward propagation
        l0 = np.array([x]).T
        l1 = tanh(np.dot(l0, syn0))
        l2 = tanh(np.dot(l1, syn1))
        l3 = np.dot(l2, syn2)

        # calc error
        if y is not None:
            l3_error = l3 - np.array([y]).T
            print('Error:' + str(np.mean(np.abs(l3_error))))

        return l3
