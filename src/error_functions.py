"""
file:   error_functions.py
author: John Maxwell
tldr:   Functions for measuring error
"""

import numpy as np


def accuracy(predicted, actual, stratified=False, num_classes=2, top=1):
    '''
    correct = 0.
    correct_num = [0.] * num_classes
    if stratified is True:
        total_num = [actual.count(i) for i in range(10)]

    for a_i, p_i in zip(actual, predicted):
        if a_i in p_i[0:top]:
            correct += 1
            correct_num[a_i] += 1

    if stratified is True:
        return {i: n / d for n, d, i in zip(correct_num, total_num, range(10))}
    else:
        return correct / len(actual)
    '''

    predicted_index = np.argmax(predicted, axis=1)
    accuracy_percent = np.sum(actual[np.arange(len(actual)), predicted_index]) / predicted.shape[0]
    return accuracy_percent


def logloss(predicted, actual):
    actual_index = np.argmax(actual, axis=1)
    log_loss = np.sum([-1 * np.log(i[j]) for i, j in zip(predicted, actual_index)]) / predicted.shape[0]
    return log_loss
