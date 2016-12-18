"""
file:   units.py
author: John Maxwell
tldr:   Activation functions and their derivatives for back propagation
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


#def tanh_activation(x, deriv=False):
#    if deriv:
#        return 1 - x ** 2
#    return (2 / (1 + np.exp(-2 * x))) - 1


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_prime(x):
    return 1 - tanh(x) ** 2


def linear(x):
    return x


def linear_prime(x):
    return 1


def softmax(x):
    out = []
    for i in x:
        out.append(i/np.sum(x))
    return out


def softmax_prime(x):
    return
