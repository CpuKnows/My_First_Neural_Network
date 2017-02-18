"""
file:   error_functions.py
author: John Maxwell
tldr:   Functions for measuring error
"""

import numpy as np


def accuracy(predicted, actual, top=1, stratified=False):
    correct = 0.
    correct_num = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    total_num = [actual.count(i) for i in range(10)]

    for a_i, p_i in zip(actual, predicted):
        if a_i in p_i[0:top]:
            correct += 1
            correct_num[a_i] += 1

    if stratified is False:
        return correct / len(actual)
    else:
        return {i: n / d for n, d, i in zip(correct_num, total_num, range(10))}


def logloss(predicted, actual):
    actual_index = np.argmax(actual, axis=1)
    log_loss = np.sum([-1 * np.log(i[j]) for i, j in zip(predicted, actual_index)]) / predicted.shape[0]
    return log_loss
