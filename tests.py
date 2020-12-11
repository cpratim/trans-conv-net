from time import time
import numpy as np
from controls import *
import os

data_dir = '../data/minute'
dates = sorted(os.listdir(f'{data_dir}/AAPL'))
symbols = read_json('../data/symbols.json')


dt = dates[-5:]

def fun1():
    channels = [[]] * 4
    for d in dt:
        bars = read_bin(f'{data_dir}/AAPL/{d}')
        _chan = [
            [b[i] for b in bars] for i in range(4)
        ]
        channels += _chan
    print(channels)

def fun2():
    channels = None
    for d in dt:
        bars = np.array(read_bin(f'{data_dir}/AAPL/{d}'))
        _chan = bars.T
        if channels is None:
            channels = np.array([_chan,])
        else:
            channels = np.concatenate((channels, _chan))

    print(channels)


def time_fun(fun):

    s = time()
    fun()
    e = time()
    return e - s

'''
t1 = time_fun(fun1)
t2 = time_fun(fun2)

print(t1)
print(t2)
'''
t1 = time_fun(fun1)
print(t1)