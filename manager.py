import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from config import *
from Src import GRU
from Src import LSTM
from Src import MLP

PATH_GOOGLE_TRACE = Config.DATASET_1_DIR
PATH_AZURE = Config.DATASET_2_DIR
FEATURE = Config.FEATURE
RATIO_TRAIN_TEST = Config.RATIO_TRAIN_TEST
RATIO_TRAIN_VAL=Config.RATIO_TRAIN_VAL


class Visualization:
    def __init__(self, path, feature):
        self.path = path
        self.index = feature
        self.data = pd.read_csv(self.path, header=None, parse_dates=True, squeeze=True)

    def line_plot(self):
        data = self.data
        plt.subplot(1, 2, 1)
        plt.plot(data.iloc[:, 0], data.iloc[:, 3])
        plt.title("CPU ")
        plt.xlabel("time")
        plt.ylabel("Mean CPU usage")
        plt.subplot(1, 2, 2)
        plt.plot(data.iloc[:, 0], data.iloc[:, 4])
        plt.title("RAM")
        plt.xlabel("time")
        plt.ylabel("Canonical Memory usage")
        plt.show()

    def ditribution(self):
        data = self.data
        plt.subplot(2, 2, 2)
        plt.title("CPU Distribution")
        data.iloc[:, 3].plot.kde(color="red")
        plt.subplot(2, 2, 1)
        plt.title("CPU Distribution")
        data.iloc[:, 3].hist()
        plt.subplot(2, 2, 3)
        plt.title("RAM Distribution")
        data.iloc[:, 4].hist()
        plt.subplot(2, 2, 4)
        plt.title("RAM Distribution")
        data.iloc[:, 4].plot.kde(color="red")
        plt.show()

    def scatter(self):
        data = self.data
        plt.subplot(1, 2, 1)
        plt.plot(data.iloc[:, 0], data.iloc[:, 3], 'go', color='blue')
        plt.xlabel("time")
        plt.ylabel("CPU")
        plt.subplot(1, 2, 2)
        plt.plot(data.iloc[:, 0], data.iloc[:, 4], "ro", color="red")
        plt.xlabel("time")
        plt.ylabel("RAM")
        plt.show()

class Preprocess:

    def __init__(self, path, feature, ratio_train_test, ratio_train_val):
        self.path = path
        self.index = feature
        self.ratio_train_test = ratio_train_test
        self.ratio_train_val = ratio_train_val
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

    def split_gru_lstm(self):
        data, scaler = self.scale()
        n = int(data.shape[0] * self.ratio_train_test)
        k = int(n * self.ratio_train_val)
        train = data.iloc[:k, :]
        val = data.iloc[k:n, :]
        test = data.iloc[n:, :]
        return val, train, test
    def split_mlp(self):
        data= pd.DataFrame(self.choice_feature())
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
        for i in range(data.shape[0] - lock_back):
            a = data[i:i + lock_back, :]
            b = data[i + lock_back, :]
            dataX.append(a)
            dataY.append(b)
        return np.array(dataX), np.array(dataY)

a=Preprocess(PATH_GOOGLE_TRACE,FEATURE,RATIO_TRAIN_TEST,RATIO_TRAIN_VAL)
def run_gru_lstm (al,a,neurol,nb_epochs,batch_size ):
    X_train,Y_train,X_test,Y_test,X_val,Y_val=al.input_data(a)
    model=al.fit_model(X_train, Y_train, X_val, Y_val, neurol, batch_size, nb_epochs)
    df,scaler=a.scale()
    al.accuracy(X_test,Y_test,scaler,model)
def run_mlp(al,a):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = al.input_data(a)
    model = al.fit_model(X_train,Y_train,X_val,Y_val)
    al.accuracy(X_test,Y_test,model)
    print (X_train)
    print (Y_train)


run_gru_lstm(GRU,a,GRU.NEUROL,GRU.EPOCHS,GRU.BATCH_SIZE)