from controls import *
import os
from indicators import Indicators
from gplearn.genetic import (
    SymbolicRegressor,
    SymbolicTransformer,
)
from gplearn.fitness import make_fitness
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data_dir = '../data/minute'

dates = sorted(os.listdir(f'{data_dir}/AAPL'))
symbols = read_json('../data/symbols.json')


def remove_outliers(x, y, out_tol=1.5):

    nX, nY = [], []
    q1, q3 = np.percentile(sorted(y), [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (out_tol * iqr)
    upper_bound = q3 + (out_tol * iqr)

    for i in range(len(y)):
        if y[i] < upper_bound and y[i] > lower_bound:
            nX.append(x[i])
            nY.append(y[i])
    return nX, nY
    

def signalize_(dt, symbols, channels, target_size, period, offset=20):
    X, y = [], []
    for sym in symbols:
        for i in range(period, len(dt)):
            try:
                dts = dt[i-period:i]
                bars = []
                for d in dts:
                    _bars = read_bin(f'{data_dir}/{sym}/{d}')
                    _offset = None
                    if d == dts[-1]:
                        (o, c) = (
                            _bars[offset+1][1],
                            _bars[-1][1],
                        )
                        
                        _offset = offset

                    bars.extend(_bars[:_offset])
                inp = []
                misf = False
                for fun, chan in channels:
                    args = [
                        [b[c] for b in bars] for c in chan
                    ]
                    _x = fun(*args)
                    if len(_x) != target_size:
                        misf = True
                        break
                    inp.append(_x)
                if misf == False:
                    y.append(1 + (c - o)/o)
                    X.append(inp)
            except Exception as e:
                pass
    X, y = remove_outliers(X, y)
    return np.array(X), np.array(y)


def signalize(dt, symbols, channels, period, off):
    X, y = [[]]*len(channels), []
    for sym in symbols:
        _channels = {0: [], 1: [], 2: [], 3: []}
        for d in dt:
            bars = read_bin(f'{data_dir}/{sym}/{d}')
            fl = 390 - len(bars)
            for i in range(4):
                _c = [b[i] for b in bars]
                if fl != 0:
                    _c += [np.average(_c) for a in range(fl)]
                _channels[i].append(_c)

        ts = []
        for k in _channels:
            ts.append(_channels[k])
        ts = np.array(ts)
        for i in range(period, len(ts[0])):
            _ts = ts[i-period:i]
            y.append( 
                1 + (_ts[1][-1][-1] - _ts[1][-1][off+1])/(_ts[1][-1][off+1])
            )
            tl = (period - 1) * 390 + off
            for i in range(len(channels)):
                fun, cn = channels[i]
                ser = [_ts[:tl, c].reshape(-1) for c in cn]
                _x = fun(*ser)
                X[i].append(_x)
    return np.array(X), np.array(y)


#vochl
#prices, rsi

(short_period, long_period, period) = (14, 28, 3)
off = 0
dt = dates[-5:]

target_size = (period - 1) * 390 + off - long_period 

ind = Indicators(
    period=short_period,
    long_period=long_period,
    target_size=target_size,
)

channels = (
    (ind.norf, (0,)),
    (ind.norf, (1,)),
    (ind.rsi, (1,)),
    (ind.sto, (1,)),
)

dir_ = 'airlines'
if not os.path.exists(dir_):
    os.mkdir(dir_)


sym = ['AAL']



X, y = signalize(
    dt=dt, 
    symbols=sym,    
    period=period,
    channels=channels,
    off=off,
)



print(X.shape)










'''
X = X.reshape(-1, target_size)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=.2,
    shuffle=False,
)


function_set = [
    'add', 'sub', 'mul', 
    'div', 'sqrt', 'log', 
    'abs', 'neg', 'inv',
    'max', 'min',
]


reg = SymbolicTransformer(
    population_size=20000,
    generations=25, 
    verbose=1,
    random_state=0,
    n_jobs=-1,
    n_components=225,
    hall_of_fame=300,
    function_set=function_set,
)
print(len(X_train))
print(len(X_test))

reg.fit(X_train, y_train)

print(reg.get_params())
trans_x_test = reg.transform(X_test)
trans_x_train = reg.transform(X_train)

x1 = trans_x_test.T[-1]
x2 = trans_x_train.T[-1]
x3 = trans_x_train.T[0]
x4 = trans_x_test.T[0]


dump_bin(f'{dir_}/train_x', trans_x_train)
dump_bin(f'{dir_}/train_y', y_train)
dump_bin(f'{dir_}/test_x', trans_x_test)
dump_bin(f'{dir_}/test_y', y_test)


print(np.corrcoef(x1, y_test))
print(np.corrcoef(x2, y_train))
print(np.corrcoef(x3, y_train))
print(np.corrcoef(x4, y_test))
'''