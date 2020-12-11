import numpy as np
from controls import *
import os
import warnings
from sklearn.cluster import (
    KMeans, 
    SpectralClustering,
    MeanShift,
)
from math import isnan
from sklearn.preprocessing import StandardScaler, normalize


warnings.filterwarnings('ignore')

data_dir = "../data/minute"
symbols = read_json('../data/symbols.json')
dates = sorted(os.listdir(f'{data_dir}/AAPL'))

def nor(a):
    if len(a) == 0:
        return None
    r = []
    for i in range(len(a)):
        r.append(a[i] / a[0])
    return r

def bstats(sym, dt, of):
    X = []
    for d in dt:
        try:
            _bars = np.array(read_bin(f'{data_dir}/{sym}/{d}')) 
            _prices, _vols = _bars[:of, 1], _bars[:of, 0]
            X.extend(_prices)
        except:
            X.extend([np.average(X) for i in range(of)])
    
    return np.array(X).reshape(-1, 1)

def signalize(dt, of=100):
    X, y = [], []
    ts = len(dt) * of
    for sym in symbols:
        x = bstats(sym, dt, of)
        x = StandardScaler().fit_transform(x).reshape(-1)
        if not np.isnan(np.min(x)):
            X.append(x)
            y.append(sym)
    return np.array(X), np.array(y)



dt = dates[-5:]
X, y = signalize(dt)

kmeans = KMeans(
    n_clusters=50,
    n_jobs=-1,
    copy_x=False,
    verbose=1,
)

mean_shift = MeanShift(
    bandwidth=4.5,
    n_jobs=-1,
)

kmeans.fit(X)
mean_shift.fit(X)
clusters = {}

labels = mean_shift.labels_
for i in range(len(labels)):
    l = labels[i]
    if l in clusters:
        clusters[l].append(y[i])
    else:
        clusters[l] = [y[i], ]

for g in clusters:
    clus = clusters[g]
    if len(clus) >= 5:
        print(clus)
