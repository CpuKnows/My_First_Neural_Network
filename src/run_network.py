"""
file:   nn.py
author: John Maxwell
tldr:   Network for forward and backward propagation
"""

from __future__ import print_function
import numpy as np

from nn_with_momentum import Network


np.random.seed(42)

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

nn = Network()

weights = nn.mini_batch_learning(x, y, n_epoch=10000, batch_size=10, learning_rate=0.01, print_error=True)

predictions = nn.forward_prop(x, weights, y=y)
print('Error:', np.mean(np.abs(predictions - y)))
