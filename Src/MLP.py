import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from config import *

EPOCHS=Config.MLP['epochs']
LOCK_BACK = Config.MLP['lock_back']

def accuracy (X,Y,model):
    Y_predict = model.predict(X)
    a = np.arange(X.shape[0])
    print("Y_predict :", Y_predict)
    print("Y ", Y)
    print("error :", sqrt(mean_squared_error(Y_predict,Y)))
    plt.plot(a, Y_predict)
    plt.plot(a, Y)
    plt.show()


def fit_model(X_train,Y_train,X_val , Y_val):
    model = Sequential()
    model.add(Dense(Config.MLP['unit1'], input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(Config.MLP['unit2'],activation='relu'))
    model.add(Dense(Config.MLP['unit3'],activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=EPOCHS, verbose=2, validation_data=(X_val, Y_val),shuffle=False)
    return model

def input_data(a):
    val, train, test = a.split_mlp()
    X_train, Y_train = a.windows_sliding(LOCK_BACK, train)
    X_val, Y_val = a.windows_sliding(LOCK_BACK, val)
    X_test, Y_test = a.windows_sliding(LOCK_BACK, test)
    n_input = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], n_input)
    X_val = X_val.reshape(X_val.shape[0], n_input)
    X_test = X_test.reshape(X_test.shape[0], n_input)
    return X_train,Y_train,X_val,Y_val,X_test,Y_test







