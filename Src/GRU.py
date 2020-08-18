import numpy as np
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from math import sqrt

PATH_GOOGLE_TRACE = 'C:/Users/ThinkKING/OneDrive/Desktop/Github/pretraining_auto_scaling_ng/data/input_data/google_trace/1_job/'
PATH_AZURE = 'C:/Users/ThinkKING/OneDrive/Desktop/Github/pretraining_auto_scaling_ng/data/input_data/azure/'
FILE="3_mins.csv"
NEUROL = 4
BATCH_SIZE = 49
EPOCHS = 3000
FEATURE = [3]
RATIO = 0.75

class Preprocess:

    def __init__(self, path, file, feature, ratio):
        self.path = path + file
        self.index = feature
        self.ratio = ratio
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
        n = data.shape[0] * self.ratio
        n = int(n)-1
        train = data.iloc[:n, :]
        test = data.iloc[n:, :]
        return train, test

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


def fit_lstm(x_train, y_train, neurol, batch_size, nb_epochs):
    X = x_train
    y = y_train
    model = Sequential()
    model.add(GRU(neurol, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()
    return model

a = Preprocess(PATH_GOOGLE_TRACE, FILE, FEATURE, RATIO)
train, test = a.split()
df, scaler = a.scale()
X_train, Y_train = a.windows_sliding(2, train)
X_test, Y_test = a.windows_sliding(2, test)
model = fit_lstm(X_train, Y_train, NEUROL, BATCH_SIZE, EPOCHS)
Y_predict = model.predict(X_test)
a = np.arange(Y_test.shape[0])

Y_predict=scaler.inverse_transform(Y_predict)
Y_test=scaler.inverse_transform(Y_test)
print ("Predict :",Y_predict)
print ("Values: ",Y_test)
plt.plot(a,Y_predict)
plt.plot(a,Y_test)
plt.show()
