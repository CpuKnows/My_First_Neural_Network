"""
file:   sigmoid_layer.py
author: John Maxwell
tldr:   Sigmoid Layer
"""

from __future__ import print_function

import numpy as np
from layer import Layer


class SoftmaxLoglossLayer(Layer):

    def __init__(self, layer_size, layer_type, below=None, above=None):
        """
        :param layer_size: number of nodes in layer
        :param below: object of class layer below in network
        :param above: object of class layer above in network
        """
        super(Layer, self).__init__()
        self.layer_size = layer_size
        self.layer_type = layer_type
        self.below = below
        self.above = above
        self.weights = None
        self.signal_values = None
        self.activation_values = None

    def forward_prop(self, x):
        """
        Forward propagation of node values x by weights w.

        :param x: forward propagation of previous layer
        :return: forward propagation of this layer
        """

        self.signal_values = x
        self.activation_values = self.softmax(self.signal_values)
        return self.activation_values

    def backward_prop(self, layer_delta):
        """
        Backward propagation of the above layers error gradient.

        :param layer_delta: gradient change of above layer
        :return: error and gradient change of this layer
        """

        y_index = np.argmax(layer_delta, axis=1)
        layer_error = np.sum([-1 * np.nan_to_num(np.log(i[j])) for i, j in zip(layer_delta, y_index)]) / \
                      layer_delta.shape[0]

        temp_delta = np.copy(self.activation_values)
        temp_delta[range(layer_delta.shape[0]), np.argmax(layer_delta, axis=1)] -= 1

        return layer_error, temp_delta

    def softmax(self, x):
        # Shift to prevent problems with large values of x
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def softmax_prime(self, x):
        pass
