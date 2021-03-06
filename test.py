import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, TimeDistributedDense
from keras.layers import LSTM, Activation
from sklearn.preprocessing import MinMaxScaler
from help_functions import *
import pydot
import graphviz
from keras.utils.visualize_util import plot


set_name= "close"

training_setX, training_setY, training_set_date = \
    generate_set("/home/adminz/repo/mgr/sets/" + set_name + "_training_set.csv",
                 "/home/adminz/repo/mgr/sets/" + set_name + "_EURUSD.csv", (0, 1))

validation_setX, validation_setY, val_date = \
    generate_set("/home/adminz/repo/mgr/sets/" + set_name + "_validation_set.csv",
                 "/home/adminz/repo/mgr/sets/" + set_name + "_EURUSD.csv", (0, 1))

test_setX, test_setY, test_set_date = \
    generate_set("/home/adminz/repo/mgr/sets/" + set_name + "_test_set.csv",
                 "/home/adminz/repo/mgr/sets/" + set_name + "_EURUSD.csv", (0, 1))

#X = numpy.reshape(training_setX, (training_setX.shape[0], 1, training_setX.shape[1]))
#Y = numpy.reshape(training_setY, (1, training_setY.shape[0]))
#x_train = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
#y_train =np.array([0, 1, 1, 0])

trainX = numpy.reshape(training_setX, (training_setX.shape[0],1, training_setX.shape[1]))
trainY= training_setY
model = Sequential([
    #TimeDistributedDense(10,input_dim=5),
    #Dense(3,input_dim=5),
    LSTM(6,input_dim=5,return_sequences=True),
    LSTM(9,input_dim=5),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=1000, batch_size=1,verbose=1)
print (model.predict(trainX))
plot(model, to_file='/home/adminz/repo/mgr/model.png')