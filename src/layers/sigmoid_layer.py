"""
file:   sigmoid_layer.py
author: John Maxwell
tldr:   Sigmoid Layer
"""

from __future__ import print_function

import numpy as np
from layer import Layer


class SigmoidLayer(Layer):

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
        self.activation_values = self.sigmoid(x)
        if self.layer_type is not 'output':
            return np.dot(self.activation_values, self.weights)
        else:
            return self.activation_values

    def backward_prop(self, layer_delta):
        """
        Backward propagation of the above layers error gradient.

        :param layer_delta: gradient change of above layer
        :return: error and gradient change of this layer
        """

        if self.layer_type is 'output':
            layer_error = self.activation_values - layer_delta
        else:
            layer_error = np.dot(layer_delta, self.weights.T)
        layer_delta = layer_error * self.sigmoid_prime(self.signal_values)

        return layer_error, layer_delta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
