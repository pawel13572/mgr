import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM, Activation
from sklearn.preprocessing import MinMaxScaler
from help_functions import *

class reccurentNetwork:
    global sets, model

    def generate_sets(self,set_path,set_name):
        global sets
        sets = generate_sets(set_path, set_name)
        return sets

    def train_network(self,epochs=1):
        global model
        trX = numpy.reshape(sets[0], (sets[0].shape[0], 1, sets[0].shape[1])) # X
        trY = sets[1] # Y
        model = Sequential([
            # TimeDistributedDense(10,input_dim=5),
            # Dense(3,input_dim=5),
            LSTM(6, input_dim=sets[0].shape[1], return_sequences=True),
            LSTM(9, input_dim=5),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trX, trY, nb_epoch=epochs, batch_size=1, verbose=1)

    def test_network(self):
        trX = numpy.reshape(sets[0], (sets[0].shape[0], 1, sets[0].shape[1]))
        print(model.predict(trX))

