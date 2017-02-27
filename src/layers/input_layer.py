"""
file:   layer.py
author: John Maxwell
tldr:   Input Layers
"""

from __future__ import print_function
import numpy as np

from layer import Layer


class InputLayer(Layer):

    def __init__(self, layer_size, layer_type, bottom=None, top=None):
        """
        :param layer_size: number of nodes in layer
        :param layer_type: input/hidden/output
        :param bottom: layer object below in network
        :param top: layer object above in network
        """
        super(Layer, self).__init__()
        self.layer_size = layer_size
        assert layer_type is 'input'
        self.layer_type = layer_type
        self.bottom = bottom
        self.top = top

    def forward_prop(self, x):
        """
        Forward propagation of node values x by weights w.

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
        layer_error = None
        # layer_delta = None

        return layer_error, layer_delta
