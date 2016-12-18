"""
file:   layer.py
author: John Maxwell
tldr:   Layers
"""

from __future__ import print_function
import numpy as np


class Layer(object):

    def __init__(self, layer_type, layer_size, activation=None, activation_prime=None, cost_function=None,
                 below=None, above=None):
        """
        :param layer_type: input, output, or hidden
        :param layer_size: number of nodes in layer
        :param activation: activation function (linear, sigmoid, tanh, relu)
        :param activation_prime: derivative of activation function (linear, sigmoid, tanh, relu)
        :param cost_function: cost function if an output layer
        :param below: object of class layer below in network
        :param above: object of class layer above in network
        """
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.node_values = None
        self.weights = None

        if self.layer_type == 'input':
            self.activation = None
            self.activation_prime = None
            self.cost_function = None
            self.below = None
            self.above = above
        elif self.layer_type == 'output':
            self.activation = activation
            self.activation_prime = activation_prime
            self.cost_function = cost_function
            self.below = below
            self.above = None
        else:
            self.activation = activation
            self.activation_prime = activation_prime
            self.cost_function = None
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

        if self.layer_type == 'input':
            self.node_values = x
            return np.dot(self.node_values, self.weights)
        elif self.layer_type == 'output':
            self.node_values = self.activation(x)
            return self.node_values
        else:
            self.node_values = self.activation(x)
            return np.dot(self.node_values, self.weights)

    def backward_prop(self, layer_delta, cost_function=None):
        """
        Backward propagation of the above layers error gradient.

        :param layer_delta: gradient change of above layer
        :param cost_function: cost function of network. Quadratic for regression, cross-entropy for classification
        :return: error and gradient change of this layer
        """
        if self.layer_type == 'output':
            # In output layer_delta == y
            #layer_error = cost_function(self.node_values, layer_delta)
            layer_error = self.node_values - layer_delta
            layer_delta = layer_error * self.activation_prime(self.node_values)
        elif self.layer_type == 'input':
            layer_error = None
        else:
            layer_error = np.dot(layer_delta, self.weights.T)
            layer_delta = layer_error * self.activation_prime(self.node_values)

        return layer_error, layer_delta
