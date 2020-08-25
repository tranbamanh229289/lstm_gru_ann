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

class Visualization :
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

#MLP.accuracy(MLP.X_test,MLP.Y_test,MLP.model)
#LSTM.accuracy(LSTM.X_test,LSTM.Y_test,LSTM.scaler,LSTM.model)
GRU.accuracy(GRU.X_test,GRU.Y_test,GRU.scaler,GRU.model)
