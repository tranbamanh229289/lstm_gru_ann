import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from config import *

PATH_GOOGLE_TRACE = Config.DATASET_1_DIR
PATH_AZURE = Config.DATASET_2_DIR

NEUROL = Config.GRU['neurol']
BATCH_SIZE = Config.GRU['batch_size']
EPOCHS = Config.GRU['epochs']
FEATURE = Config.FEATURE
RATIO_TRAIN_TEST = Config.RATIO_TRAIN_TEST
RATIO_TRAIN_VAL=Config.RATIO_TRAIN_VAL

class Preprocess:

    def __init__(self, path, feature, ratio_train_test,ratio_train_val):
        self.path = path
        self.index = feature
        self.ratio_train_test = ratio_train_test
        self.ratio_train_val=ratio_train_val
        self.data = pd.read_csv(self.path, header=None, parse_dates=True, squeeze=True)

    def choice_feature(self):
        data = self.data
        df = data.iloc[:, self.index]
        return df

    def scale(self):
        data = self.choice_feature()
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(data)
        return pd.DataFrame(dataset), scaler

    def split(self):
        data, scaler = self.scale()
        n = int(data.shape[0] * self.ratio_train_test)
        k = int(n * self.ratio_train_val)
        train = data.iloc[:k, :]
        val = data.iloc[k:n, :]
        test = data.iloc[n:, :]
        return val, train, test

    def windows_sliding(self, lock_back, data):
        data = data.values
        dataX = []
        dataY = []
        for i in range(data.shape[0] - lock_back ):
            a = data[i:i + lock_back, :]
            b = data[i + lock_back, :]
            dataX.append(a)
            dataY.append(b)
        return np.array(dataX), np.array(dataY)


def fit_gru(x_train, y_train,x_val,y_val, neurol, batch_size, nb_epochs):
    X = x_train
    y = y_train
    model = Sequential()
    model.add(GRU(neurol, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size,validation_data=(x_val,y_val), verbose=2, shuffle=False)
        model.reset_states()
    return model

def accuracy (X,Y, scaler,model):
    Y_predict = model.predict(X)
    a = np.arange(X.shape[0])
    Y_predict = scaler.inverse_transform(Y_predict)
    Y = scaler.inverse_transform(Y)
    print("Y_predict :", Y_predict)
    print("Y ", Y)
    print("error :", model.evaluate(X, Y, verbose=0))
    plt.plot(a, Y_predict)
    plt.plot(a, Y)
    plt.show()

a = Preprocess(PATH_GOOGLE_TRACE, FEATURE, RATIO_TRAIN_TEST,RATIO_TRAIN_VAL)
val,train, test= a.split()
df, scaler = a.scale()
X_train, Y_train = a.windows_sliding(Config.GRU['lock_back'], train)
X_test, Y_test = a.windows_sliding(Config.GRU['lock_back'], test)
X_val,Y_val = a.windows_sliding(Config.GRU['lock_back'],val)
model = fit_gru(X_train, Y_train,X_val,Y_val, NEUROL, BATCH_SIZE, EPOCHS)
