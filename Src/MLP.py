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

PATH_GOOGLE_TRACE = Config.DATASET_1_DIR
PATH_AZURE = Config.DATASET_2_DIR
FEATURE = Config.FEATURE
RATIO_TRAIN_TEST = Config.RATIO_TRAIN_TEST
RATIO_TRAIN_VAL=Config.RATIO_TRAIN_VAL
EPOCHS=Config.MLP['epochs']
LOCK_BACK = Config.MLP['lock_back']

class Preprocess:

    def __init__(self, path, feature, ratio_train_test,ratio_train_val):
        self.path = path
        self.index = feature
        self.ratio_train_test = ratio_train_test
        self.ratio_train_val = ratio_train_val
        self.data = pd.read_csv(self.path, header=None, parse_dates=True, squeeze=True)

    def choice_feature(self):
        data = self.data
        df = data.iloc[:, self.index]
        return df

    def split(self):
        data= pd.DataFrame(self.choice_feature())
        n = int (data.shape[0] * self.ratio_train_test)
        k=int (n*self.ratio_train_val)
        train = data.iloc[:k, :]
        val = data.iloc[k:n,:]
        test = data.iloc[n:, :]
        return val , train, test

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

def accuracy (X,Y,model):
    Y_predict = model.predict(X)
    a = np.arange(X.shape[0])
    print("Y_predict :", Y_predict)
    print("Y ", Y)
    print("error :", sqrt(mean_squared_error(Y_test,Y_predict)))
    plt.plot(a, Y_predict)
    plt.plot(a, Y)
    plt.show()


a= Preprocess(PATH_GOOGLE_TRACE,FEATURE, RATIO_TRAIN_TEST, RATIO_TRAIN_VAL)
val, train , test = a.split()
X_train , Y_train= a.windows_sliding(LOCK_BACK,train)
X_val , Y_val = a.windows_sliding(LOCK_BACK,val)
X_test,Y_test=a.windows_sliding(LOCK_BACK,test)

n_input= X_train.shape[1]*X_train.shape[2]
X_train=X_train.reshape(X_train.shape[0],n_input)
X_val = X_val .reshape(X_val.shape[0],n_input)
X_test=X_test.reshape(X_test.shape[0],n_input)
model = Sequential()
model.add(Dense(Config.MLP['unit1'],input_dim=LOCK_BACK))
model.add(Dense(Config.MLP['unit2']))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,Y_train,epochs=EPOCHS,verbose=2,validation_data=(X_val,Y_val))

