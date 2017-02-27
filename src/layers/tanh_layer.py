"""
file:   tanh_layer.py
author: John Maxwell
tldr:   TanH Layer
"""

from __future__ import print_function

import numpy as np
from layer import Layer


class TanhLayer(Layer):

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
        self.activation_values = self.tanh(self.signal_values)
        return np.dot(self.activation_values, self.weights)

    def backward_prop(self, layer_delta):
        """
        Backward propagation of the above layers error gradient.

        :param layer_delta: gradient change of above layer
        :return: error and gradient change of this layer
        """

        if self.layer_type is 'output':
            layer_error = layer_delta - self.activation_values
        else:
            layer_error = np.dot(layer_delta, self.weights.T)
        layer_delta = layer_error * self.tanh_prime(self.signal_values)

        return layer_error, layer_delta

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def tanh_prime(self, x):
        return 1 - self.tanh(x) ** 2
