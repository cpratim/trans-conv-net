from gplearn.functions import make_function
import numpy as np
import scipy.stats as stats

def _exp(x1, x2):
    return x1 ** x2

def _greater_than(x1, x2):
    if np.mean(x1) > np.mean(x2):
        return x1
    return x2

def _less_than(x1, x2):

    if np.mean(x1) < np.mean(x2):
        return x1
    return x2

def _zscore(x1):
    for i in x1:
        if i <= 0:
            return x1
    return stats.zscore(x1)

exp = make_function(
    function=_exp,
    name='exp',
    arity=2,
)

zscore = make_function(
    function=_zscore,
    name='zscore',
    arity=1,
)

greater_than = make_function(
    function=_greater_than,
    name='greater_than',
    arity=2,
)

less_than = make_function(
    function=_less_than,
    name='less_than',
    arity=2,
)