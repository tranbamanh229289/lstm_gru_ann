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

NEUROL = Config.LSTM['neurol']
BATCH_SIZE = Config.LSTM['batch_size']
EPOCHS = Config.LSTM['epochs']

def fit_model(x_train, y_train, x_val, y_val, neurol, batch_size, nb_epochs):
    X = x_train
    y = y_train
    model = Sequential()
    model.add(LSTM(neurol, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    for i in range(nb_epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size, validation_data=(x_val, y_val), verbose=2, shuffle=False)
        model.reset_states()
    return model

def accuracy(X, Y, scaler, model):
    Y_predict = model.predict(X)
    a = np.arange(X.shape[0])
    Y_predict = scaler.inverse_transform(Y_predict)
    Y = scaler.inverse_transform(Y)
    #print("Y_predict :", Y_predict)
    #print("Y ", Y)
    print("error :", sqrt(mean_squared_error(Y, Y_predict)))
    plt.plot(a, Y_predict)
    plt.plot(a, Y)
    plt.show()

def input_data(a):
    val, train, test = a.split_gru_lstm()
    X_train, Y_train = a.windows_sliding(Config.GRU['lock_back'], train)
    X_test, Y_test = a.windows_sliding(Config.GRU['lock_back'], test)
    X_val, Y_val = a.windows_sliding(Config.GRU['lock_back'], val)
    return X_train,Y_train,X_test,Y_test,X_val,Y_val




