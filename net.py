from controls import *
from gplearn.genetic import (
    SymbolicRegressor,
    SymbolicTransformer,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier
)
from sklearn.neighbors import (
    KNeighborsRegressor
)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from helpers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from syn import SyntheticRegressor


def load_dataset(
        feat_files, 
        target_file, 
        test_size=.2, 
        shuffle=False, 
        tts=False,
        classification=False
    ):

    target = read_bin(target_file)
    features = [
        read_bin(f) for f in feat_files
    ]

    if classification == False:
        y = target
    else:
        y = []
        for i in target:
            if i > 0:
                y.append(1)
            else:
                y.append(0)

    X = []
    for i in range(len(features[0])):
        _feats = [
            f[i] for f in features 
        ]
        X.append(
            np.concatenate(_feats)
        )

    if tts:
        ret = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=10,
            shuffle=shuffle,
        )
    else:
        ret = (X, y)
    return (
        np.array(a) for a in ret
    )

def load_dataframe(feat_files, target_file, classification=False, save=False, filename=None):
    target = read_bin(target_file)
    features = [
        read_bin(f) for f in feat_files
    ]
    if classification == False:
        y = target
    else:
        y = []
        for i in target:
            if i > 0:
                y.append(1)
            else:
                y.append(0)
    X = []
    samples = len(features[0])
    for i in range(samples):
        _feats = [
            f[i] for f in features 
        ] + [[y[i]]]
        X.append(
            np.concatenate(_feats)
        )

    columns = [
        f'feat_{i}' for i in range(1, len(X[0]))
    ] + ['target']

    df = pd.DataFrame(
        X,
        columns=columns,
    )
    
    if save:
        df.to_csv(filename, index = False) 
    return df

train_feat_files =(
    'AAL/train/price',
    'AAL/train/rsi',
    'AAL/train/vol',
)
train_target_file = 'AAL/train/target'


test_feat_files =(
    'AAL/test/price',
    'AAL/test/rsi',
    'AAL/test/vol',
)
test_target_file = 'AAL/test/target'


X_train, y_train = load_dataset(
    feat_files=train_feat_files,
    target_file=train_target_file,
    tts=False,
    classification=True,
)

inp_dim = len(X_train[0])

X_test, y_test = load_dataset(
    feat_files=test_feat_files,
    target_file=test_target_file,
    tts=False,
    classification=True,
)

X_train = X_train.reshape(-1, inp_dim)
X_test = X_test.reshape(-1, inp_dim)

function_set = [
    'add', 'sub', 'mul', 
    'div', 'sqrt', 'log', 
    'abs', 'neg', 'inv',
    'max', 'min', exp,
    greater_than, less_than,
    zscore,
]


model = Sequential([
    Dense(20, input_dim=inp_dim, kernel_initializer='normal', activation='relu'),
    Dense(20, activation='tanh'),
    Dense(15, activation='tanh'),
    Dense(4, activation='relu'),
    Dense(4, activation='tanh'),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_split=0.2)

_pred = model.predict(X_test)
pred = [int(i > .5) for i in _pred]
#pred = _pred

c = 0
for i in range(len(pred)):
    print(pred[i], y_test[i])
    if pred[i] == y_test[i]:
        c += 1

print(c/len(pred))

