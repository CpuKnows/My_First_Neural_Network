"""
file:   layer.py
author: John Maxwell
tldr:   Layers
"""

from __future__ import print_function
import numpy as np


class Layer(object):

    def __init__(self, layer_size, layer_type, bottom=None, top=None):
        """
        :param layer_size: number of nodes in layer
        :param bottom: layer object below in network
        :param top: layer object above in network
        """
        self.layer_size = layer_size
        self.layer_type = layer_type
        self.bottom = bottom
        self.top = top

    def forward_prop(self, x):
        """
        Forward propagation of node values x with activation function f() by weights w.
        If input layer, then an activation function isn't used.
        If output layer, then there are no weights.

        :param x: forward propagation of previous layer
        :return: forward propagation of this layer
        """
        return self.top.forward_prop(x)

    def backward_prop(self, layer_delta):
        """
        Backward propagation of the above layers error gradient.

        :param layer_delta: gradient change of above layer
        :return: error and gradient change of this layer
        """
        if self.layer_type is 'output':
            layer_error = self.top.activation_values - layer_delta
        else:
            layer_error = self.top.backward_prop(layer_delta)
        layer_delta = layer_error

        return layer_error, layer_delta
