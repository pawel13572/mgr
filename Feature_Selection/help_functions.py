from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
import math
import numpy as np


def evaluate_errors(name_set,real,predict):
    print "____________",name_set,"set___________"
    print "Mean Square Error: ", mean_squared_error(real, predict)
    print "Root Mean Square Error: ", math.sqrt(mean_squared_error(real, predict))
    print "Median Square Error: ", median_absolute_error(real, predict)
    print "Mean Absolute Error: : ", median_absolute_error(real, predict)

def generate_set(file_name): #OSTATNIA WARTOSC WEKTORA TO OCZEKIWANY OUTPUT
    setX=pd.read_csv(file_name, header=0, usecols=[1, 2, 3, 4, 5, 6], delimiter=';')
    set = [setX['Close1'], setX['Close2'],
           setX['Close3'], setX['Close4'],
           setX['Close5'], setX['Close0']]
    return set

def generate_set_ALL(file_name): #OSTATNIA WARTOSC WEKTORA TO OCZEKIWANY OUTPUT
    setX=pd.read_csv(file_name, header=0, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], delimiter=';')
    set = [setX['Close1'], setX['Close2'],
           setX['Close3'], setX['Close4'],
           setX['Close5'], setX['MA5'],
           setX['MA10'], setX['MA20'],
           setX['MA60'], setX['MA120']]
    return set

def get_date(file_name): #OSTATNIA WARTOSC WEKTORA TO OCZEKIWANY OUTPUT
    setX=pd.read_csv(file_name, header=0, usecols=[0], delimiter=';')
    return setX['Date']

def generate_train_vector(set_name):  # OSTATNIA WARTOSC WEKTORA TO OCZEKIWANY OUTPUT
    vector=np.array(set_name[0:(len(set_name)-1)])
    #for i in range(0, len(set_name)-1):
        #for j in range(0,len(set_name[0])):
            #vector.append(set_name[j][i])
    return vector

def generate_network(input_neurons,hidden_neurons,output_neurons):
    n = FeedForwardNetwork()
    inLayer = LinearLayer(input_neurons)
    hiddenLayer = SigmoidLayer(hidden_neurons)
    outLayer = SigmoidLayer(output_neurons)

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    n.addConnection(in_to_hidden)
    n.addConnection(hidden_to_out)

    n.sortModules()
    return n