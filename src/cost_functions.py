"""
file:   cost_functions.py
author: John Maxwell
tldr:   Cost functions for regression and classification
"""

import numpy as np


class QuadraticCost(object):

    def error(self, a, y):
        return 0.5*np.linalg.norm(a-y)**2

    def delta(self, a, y):
        return a - y


class CrossEntropyCost(object):

    def error(self, a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    def delta(self, a, y):
        #return (a - y) / ((1 - a) * a)
        return a - y
