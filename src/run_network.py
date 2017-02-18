"""
file:   run_network.py
author: John Maxwell
tldr:   Network for forward and backward propagation
"""

from __future__ import print_function
import pickle
import numpy as np

from nn_with_adv_layers import Network
from layers.input_layer import InputLayer
from layers.linear_layer import LinearLayer
from layers.sigmoid_layer import SigmoidLayer
from layers.softmax_logloss_layer import SoftmaxLoglossLayer
from layers.tanh_layer import TanhLayer


np.random.seed(42)

# Load and clean data
x = None
y = None

# Split x and y into training and validation sets
x_train = None
y_train = None
x_val = None
y_val = None

# Create NN
nn = Network()

nn.add_layer(InputLayer(10))
nn.add_layer(SigmoidLayer(10))
nn.add_layer(LinearLayer(10))
nn.add_layer(TanhLayer(10))
nn.add_layer(SoftmaxLoglossLayer(10))

nn.initialize_weights()

# Train network
nn.mini_batch_learning(x_train, y_train, x_val, y_val, n_epoch=100, batch_size=10, learning_rate=0.01,
                       momentum=True, print_error=True, print_every=10)

# Predictions
predictions = nn.forward_prop(x_val)
print('Error:', np.mean(np.abs(predictions - y_val)))

# Save the network
pickle.dump(nn, open('file path here', 'wb'))
