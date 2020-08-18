from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
K = np.random.randint(0, 10000)
RATIO_VAL_TRAIN = 0.8
NUM_EPOCHS = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
N = int(x_train.shape[0] * RATIO_VAL_TRAIN)
x_val = x_train[N:, :]
y_val = y_train[N:]
x_train = x_train[:N, :]
y_train = y_train[:N]

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_val = x_val.reshape((x_val.shape[0], 28, 28, 1))

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
y_val = utils.to_categorical(y_val, 10)


def fit(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    H = model.fit(x_train, y_train, batch_size=1000, epochs=NUM_EPOCHS, validation_data=(x_val, y_val), verbose=1)
    return model, H


def accuracy_loss(H):
    plt.plot(np.arange(0, NUM_EPOCHS), H.history['loss'], label='training_loss')
    plt.plot(np.arange(0, NUM_EPOCHS), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, NUM_EPOCHS), H.history['accuracy'], label='accuracy')
    plt.plot(np.arange(0, NUM_EPOCHS), H.history['val_accuracy'], label='validation_accuracy')
    plt.xlabel('EPOCHS')
    plt.ylabel('Loss|Accuracy')
    plt.title('Loss and Accuracy of trainingset and validationset')
    plt.legend()
    plt.show()


def predict_image(x_test, y_test, model, random):
    y_predict = model.predict(x_test[random].reshape(1, 28, 28, 1))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Value predict of x_test[random] by probability : ", y_predict)
    print("Predict: ", np.argmax(y_predict))
    print("Accuracy test dataset :", score[2] * 100, "%")
    plt.imshow(x_test[random].reshape(28, 28))
    plt.show()


model, H = fit(x_train, y_train)
accuracy_loss(H)
predict_image(x_test, y_test, model, K)
