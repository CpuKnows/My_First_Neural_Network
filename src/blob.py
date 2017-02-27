"""
file:   blob.py
author: John Maxwell
tldr:   Layers
"""

from __future__ import print_function

from layer import Layer


class Blob(object):

    def __init__(self, bottom, top):
        isinstance(bottom, Layer)
        isinstance(top, Layer)

        self.bottom = bottom
        self.top = top
        self.shape = (bottom.layer_size, top.layer_size)
        self.signal_values = None
        self.weights = None
        self.activation_values = None

    def set_weights(self, weights):
        assert weights.shape == self.shape
        self.weights = weights

    def get_weights(self):
        return self.weights

    def calculate_signal(self, input_values):
        self.activation_values = input_values
        self.signal_values = self.activation_values * self.weights
        return self.signal_values
