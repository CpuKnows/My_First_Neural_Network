"""
file:   blob.py
author: John Maxwell
tldr:   Layers
"""

from __future__ import print_function
import numpy as np

from layer import Layer


class Blob(object):

    def __init__(self, bottom, top):
        self.bottom = bottom
        self.top = top

        if self.top is None:
            # Output blob
            self.shape = None
        else:
            self.shape = (bottom.layer_size, top.layer_size)

        self.signal_values = None
        self.activation_values = None

        if self.top is None:
            # Output blob
            self.weights = None
        else:
            self.weights = np.zeros(self.shape)

    def set_weights(self, weights):
        assert self.top is not None
        assert self.weights.shape == self.shape
        self.weights = weights

    def get_weights(self):
        assert self.top is not None
        return self.weights

    def forward_prop(self, activation_values):
        self.activation_values = activation_values

        if self.top is None:
            # Output blob
            return self.activation_values
        else:
            self.signal_values = np.dot(self.activation_values, self.weights)
            return self.signal_values

    def backward_prop(self, layer_delta):
        assert self.top is not None
        return np.dot(layer_delta, self.weights.T)
