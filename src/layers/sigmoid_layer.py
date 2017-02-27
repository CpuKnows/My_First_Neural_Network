"""
file:   sigmoid_layer.py
author: John Maxwell
tldr:   Sigmoid Layer
"""

from __future__ import print_function
import numpy as np

from layer import Layer


class SigmoidLayer(Layer):

    def __init__(self, layer_size, layer_type, bottom=None, top=None):
        """
        :param layer_size: number of nodes in layer
        :param layer_type: input/hidden/output
        :param bottom: layer object below in network
        :param top: layer object above in network
        """
        super(Layer, self).__init__()
        self.layer_size = layer_size
        self.layer_type = layer_type
        self.bottom = bottom
        self.top = top

    def forward_prop(self, x):
        """
        Forward propagation of node values x by weights w.

        :param x: forward propagation of previous layer
        :return: forward propagation of this layer
        """
        return self.top.forward_prop(self.sigmoid(x))

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
        layer_delta = layer_error * self.sigmoid_prime(self.bottom.signal_values)

        return layer_error, layer_delta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
