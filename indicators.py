from pyti.exponential_moving_average import exponential_moving_average as _ema
from pyti.relative_strength_index import relative_strength_index as _rsi
from pyti.triple_exponential_moving_average import triple_exponential_moving_average as _ema3
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as _macd
from pyti.ultimate_oscillator import ultimate_oscillator as _uo
from pyti.volume_oscillator import volume_oscillator as _vo
from pyti.stochastic import percent_d as _sto
import numpy as np
from math import isnan
from controls import *
import os



class Indicators(object):

    def __init__(self, period, long_period, target_size):
        self.period = period
        self.long_period = long_period
        self.target_size = target_size



    def nor(self, a):
        if len(a) == 0:
            return None
        r = []
        for i in range(len(a)):
            r.append(a[i] / a[0])
        return r

    def norf(self, a):
        return self.nor(a[self.long_period:])
        
    def clean(self, a):
        cleaned = np.nan_to_num(
            a, 
            copy=True, 
            nan=1
        )
        return self.nor(
            cleaned[self.target_size-len(cleaned):]
        )

    def ema(self, p):
        res = _ema(p, self.period)
        return self.clean(res)
        
    def rsi(self, p):
        res = _rsi(p, self.period)
        return self.clean(res)

    def ema3(self, p):
        res = _ema3(p, self.period)
        return self.clean(res)

    def macd(self, p):
        res = _macd(p, self.period, self.long_period)
        return self.clean(res)

    def vo(self, v):
        res = _vo(v, self.period, self.long_period)
        return self.clean(res)


    def uo(self, c, l):
        res = _uo(c, l)
        return self.clean(res)

    def sto(self, p):
        res = _sto(p, self.long_period)
        return self.clean(res)