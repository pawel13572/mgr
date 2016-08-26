from pybrain.structure import LinearLayer, SigmoidLayer,FullConnection,FeedForwardNetwork,LinearConnection,BiasUnit
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from numpy import genfromtxt
import datetime as dt
import pickle
import sklearn.preprocessing as sc
from pybrain.structure import RecurrentNetwork


def evaluate_errors(sets,path):
    text_file = open('D:/PycharmProjects/Forex/' + path + '_' + 'errors.txt', "w")
    error_val=0
    error_train=0
    for i in range(0,3):
        real, predict = sets[i], sets[i + 3]
        if(i==0):
            text_file.write("Training Set\n")
            error_train=mean_squared_error(real, predict)
        elif(i==1):
            text_file.write("Validation Set\n")
            error_val = mean_squared_error(real, predict)
        else:
            text_file.write("Test Set\n")

        text_file.write("Mean Square Error: %s\n" % mean_squared_error(real, predict))
        text_file.write("Root Mean Square Error: %s\n" % math.sqrt(mean_squared_error(real, predict)))
        text_file.write("Median Square Error: %s\n" % median_absolute_error(real, predict))
        text_file.write("Mean Absolute Error: %s\n" % mean_absolute_error(real, predict))
    text_file.close()
    return error_train,error_val

def scale_linear_bycolumn(rawpoints):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    return (rawpoints-mins)/(maxs-mins)

def generate_set(file_name,file_name_to_scale,delete_columns): #OSTATNIA WARTOSC WEKTORA TO OCZEKIWANY OUTPUT
    minmax = genfromtxt(file_name_to_scale, delimiter=';')
    minmax = np.delete(minmax, (0), axis=0)
    minmax = np.delete(minmax, delete_columns, axis=1)

    mins = np.min(minmax, axis=0)
    maxs = np.max(minmax, axis=0)

    my_data = genfromtxt(file_name, delimiter=';')
    my_data = np.delete(my_data, (0), axis=0)
    Y = my_data[:, 1]
    my_data = np.delete(my_data, delete_columns, axis=1)
    date = pd.read_csv(file_name, header=0, usecols=[0], delimiter=';')

    Y =(Y-mins[0])/(maxs[0]-mins[0])
    my_data = (my_data-mins)/(maxs-mins)
    return my_data, Y, date['Date']


def generate_sets(set_path,set_name):
    sets = [None] * 9
    sets[0], sets[1], sets[2] = \
        generate_set(set_path + set_name + "_training_set.csv",
                     set_path + set_name + "_EURUSD.csv", (0, 1))

    sets[3], sets[4], sets[5] = \
        generate_set(set_path + set_name + "_validation_set.csv",
                     set_path + set_name + "_EURUSD.csv", (0, 1))

    sets[6], sets[7], sets[8] = \
        generate_set(set_path + set_name + "_test_set.csv",
                     set_path + set_name + "_EURUSD.csv", (0, 1))
    return sets


def rescale(set, min, max):
    #set = sc.minmax_scale(set,feature_range=(min,max),copy=False)
    set2 = (set * (max - min)) + min
    return set2


def generate_network(input_neurons,hidden_neurons,output_neurons, network_type=0):
    if network_type==0:
        n = FeedForwardNetwork()
    else:
        n = RecurrentNetwork()


    inLayer = LinearLayer(input_neurons) #linear
    hiddenLayer = SigmoidLayer(hidden_neurons) #sigmoid
    outLayer = SigmoidLayer(output_neurons) #sigmoid
    bias = BiasUnit()

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)
    n.addModule(bias)

    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    bias_to_hidden = FullConnection(bias, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)

    n.addConnection(in_to_hidden)
    n.addConnection(bias_to_hidden)
    n.addConnection(hidden_to_out)

    if network_type==1:
        n.addRecurrentConnection(FullConnection(outLayer, hiddenLayer))
        #n.addRecurrentConnection(FullConnection(hiddenLayer, hiddenLayer))

    n.sortModules()
    return n


def generate_linear_network(input_neurons,output_neurons):
    n = FeedForwardNetwork()

    inLayer = LinearLayer(input_neurons)
    outLayer = LinearLayer(output_neurons)

    n.addInputModule(inLayer)
    n.addOutputModule(outLayer)

    in_to_out = FullConnection(inLayer, outLayer)

    n.addConnection(in_to_out)

    n.sortModules()
    return n

def save_plots(A,labelA,B,labelB,date,path,plot_name):
    x_axis = [dt.datetime.strptime(d, '%d.%m.%Y').date() for d in date]
    plt.plot(x_axis, A, label=labelA)
    plt.plot(x_axis, B, label=labelB)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.savefig('D:/PycharmProjects/Forex/'+ path+ '_' + plot_name+'.jpg', format='jpg', dpi=1200)
    plt.close()

def save_network(network,path):
    networkk = open('D:/PycharmProjects/Forex/' + path, 'w')
    pickle.dump(network, networkk)
    networkk.close()