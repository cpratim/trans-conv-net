from controls import *
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def load_dataset(
        feat_files, 
        target_file, 
        test_size=.2, 
        shuffle=False, 
        tts=False,
        classification=False,
        target_size=289,
    ):

    target = read_bin(target_file)
    features = np.array([
        read_bin(f) for f in feat_files
    ])

    
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
    for i in range(len(features)):
        _feats = features[i]
        for i in range(len(_feats)):
            _f = _feats[i]
            if len(X) == i:
                X.append(_f)
            else:
                X[i] = np.concatenate((X[i], _f))

    if tts:
        ret = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=10,
            shuffle=shuffle,
        )
    for i in range(len(X)):
        X[i] = X[i][:target_size]
    else:
        ret = (X, y)
    return (
        np.array(a) for a in ret
    )
    
dir_ = 'airlines'
train_feat_files =(
    f'{dir_}/train/rsi',
    f'{dir_}/train/vol',
    f'{dir_}/train/price',
)
train_target_file = f'{dir_}/train/target'


test_feat_files =(
    f'{dir_}/test/rsi',
    f'{dir_}/test/vol',
    f'{dir_}/test/price',
)
test_target_file = f'{dir_}/test/target'

X_train, y_train = load_dataset(
    feat_files=train_feat_files,
    target_file=train_target_file,
    tts=False,
    classification=False,
)


X_test, y_test = load_dataset(
    feat_files=test_feat_files,
    target_file=test_target_file,
    tts=False,
    classification=False,
)

print(len(X_train), len(y_train))

scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

dim1, dim2 = 17, 17
X_train = X_train.reshape(-1, dim1, dim2, 1)
X_test = X_test.reshape(-1, dim1, dim2, 1)

model = Sequential([
    Conv2D(
        30, (1, 1),
        input_shape=(dim1, dim2, 1), 
        activation='relu'
    ),
    MaxPooling2D(
        (1, 1), 
    ),
    Conv2D(
        40, (1, 1),
        activation='relu'
    ),
    MaxPooling2D(
        (1, 1), 
    ),
    Conv2D(
        50, (1, 1),
        activation='relu'
    ),
    MaxPooling2D(
        (1, 1), 
    ),
    Conv2D(
        60, (1, 1),
        activation='relu'
    ),
    MaxPooling2D(
        (1, 1), 
    ),
    Conv2D(
        40, (1, 1),
        activation='relu'
    ),
    MaxPooling2D(
        (1, 1), 
    ),
    Flatten(),
    Dense(60, activation='relu'),
    Dense(30, activation='relu'),
    Dense(15, activation='relu'),
    Dense(5, activation='relu'),
    Dense(1, activation='linear')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mse']
)

model.fit(X_train, y_train, epochs=20)

pred = model.predict(X_test)
#pred = [int(i[0] > .5) for i in pred]

dist = []
prof = 1
c = 0
for i in range(len(pred)):

    p, r = pred[i], y_test[i] 
    if (p-1)/(r-1) > 0:
        print(i, 'p', prof, abs(1 - r) + 1)
        prof *= abs(1 - r) + 1
    else:
        print(i, 'l', prof, 1 - abs(1- r))
        prof *= 1 - abs(1- r)

    dist.append(abs(pred[i] - y_test[i]))

print(np.average(dist))
print(prof)