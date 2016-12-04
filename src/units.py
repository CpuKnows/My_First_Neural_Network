"""
file:   units.py
author: John Maxwell
tldr:   Activation functions and their derivatives for back propagation
"""

import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def tanh(x, deriv=False):
    if deriv:
        return 1 - x ** 2
    return (2 / (1 + np.exp(-2 * x))) - 1
