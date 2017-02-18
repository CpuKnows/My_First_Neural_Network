"""
file:   layer.py
author: John Maxwell
tldr:   Layers
"""

from __future__ import print_function
import numpy as np


class Layer(object):

    def __init__(self, layer_size, below=None, above=None):
        """
        :param layer_size: number of nodes in layer
        :param below: object of class layer below in network
        :param above: object of class layer above in network
        """
        self.layer_size = layer_size
        self.node_values = None
        self.weights = None
        self.below = below
        self.above = above
        self.dropout = False

    def forward_prop(self, x):
        """
        Forward propagation of node values x with activation function f() by weights w.
        If input layer, then an activation function isn't used.
        If output layer, then there are no weights.

        :param x: forward propagation of previous layer
        :return: forward propagation of this layer
        """

        return np.dot(self.node_values, self.weights)

    def backward_prop(self, layer_delta):
        """
        Backward propagation of the above layers error gradient.

        :param layer_delta: gradient change of above layer
        :return: error and gradient change of this layer
        """

        layer_error = np.dot(layer_delta, self.weights.T)
        layer_delta = layer_error

        return layer_error, layer_delta
