"""
file:   sigmoid_layer.py
author: John Maxwell
tldr:   Sigmoid Layer
"""

from __future__ import print_function

import numpy as np
from layer import Layer


class SigmoidLayer(Layer):

    def __init__(self, layer_size, below=None, above=None):
        """
        :param layer_size: number of nodes in layer
        :param below: object of class layer below in network
        :param above: object of class layer above in network
        """
        super(Layer, self).__init__()
        self.layer_size = layer_size
        self.below = below
        self.above = above
        self.node_values = None
        self.weights = None
        self.dropout = False

    def forward_prop(self, x):
        """
        Forward propagation of node values x by weights w.

        :param x: forward propagation of previous layer
        :return: forward propagation of this layer
        """

        self.node_values = self.sigmoid(x)
        return np.dot(self.node_values, self.weights)

    def backward_prop(self, layer_delta):
        """
        Backward propagation of the above layers error gradient.

        :param layer_delta: gradient change of above layer
        :return: error and gradient change of this layer
        """

        layer_error = np.dot(layer_delta, self.weights.T)
        layer_delta = layer_error * self.sigmoid_prime(self.node_values)

        return layer_error, layer_delta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))