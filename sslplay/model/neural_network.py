# Original implementation by: https://github.com/divamgupta/ladder_network_keras

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler



class ModelNeuralNetwork:


    def __init__(self):
        self.name = "NEURAL-NETWORK"
        self.scaler = MinMaxScaler()


    def fit(self, X, y, Xu=None):
        self.Xl = self.scaler.fit_transform(X).reshape(X.shape[0], 1, -1)

        self.yl = y

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1, self.Xl.shape[2])),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(500, activation='relu'),
            keras.layers.Dense(250, activation='relu'),
            keras.layers.Dense(250, activation='relu'),
            keras.layers.Dense(250, activation='relu'),
            keras.layers.Dense(len(np.unique(y)), activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.fit(self.Xl, self.yl, epochs=10)


    def predict(self, X):
        self.Xt = self.scaler.transform(X).reshape(X.shape[0], 1, -1)
        return self.model.predict(self.Xt)