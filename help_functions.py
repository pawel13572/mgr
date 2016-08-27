from pybrain.structure import LinearLayer, SigmoidLayer,FullConnection,FeedForwardNetwork,LinearConnection,BiasUnit, LSTMLayer
from pybrain.datasets import SupervisedDataSet,SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
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
    text_file = open(path + '_' + 'errors.txt', "w")
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


def generate_supervised_data_set(x, y):
    ds = SupervisedDataSet(len(x[0]), 1)
    for i in range(0, len(y)):
        ds.addSample(x[i], y[i])

    return ds


def generate_sequential_data_set(x, y):
    ds = SequentialDataSet(len(x[0]), 1)
    for i in range(0, len(y)):
        ds.addSample(x[i], y[i])

    return ds


def rescale(set, min, max):
    set2 = (set * (max - min)) + min
    return set2


def generate_mlp_network(input_neurons,hidden_neurons,output_neurons):
    n = FeedForwardNetwork()

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

    n.sortModules()
    return n


def generate_rnn_network(input_neurons,hidden_neurons,output_neurons):
    n = buildNetwork(input_neurons, hidden_neurons, output_neurons, bias=True,
                           hiddenclass=LSTMLayer,
                           # hiddenclass=SigmoidLayer,
                           outclass=SigmoidLayer,
                           outputbias=False, fast=False, recurrent=True)
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


def train_network(network,data_set,epochs=10, expected_train_error=0):
    trainer = BackpropTrainer(network, dataset=data_set)
    train_errors=[] # list of all errors to plot the error chart
    actual_error = 0
    if expected_train_error != 0:
        while actual_error > expected_train_error:
            trainer.trainEpochs(epochs=1)
            actual_error = trainer.testOnData()
            train_errors.append(actual_error)
    else:
        for i in range(0,epochs):
            trainer.trainEpochs(epochs=1)
            actual_error=trainer.testOnData()
            print(i,actual_error)
            train_errors.append(actual_error)

    return network, train_errors


def generate_plots_and_errors(save_results_path, set_name, network, train_errors, sets):
    path = save_results_path+set_name

    train = []
    validate = []
    test = []
    for i in range(0, len(sets[1])):
        train.append(network.activate(sets[0][i]))  # train
        if i < len(sets[4]):
            validate.append(network.activate(sets[3][i]))  # validation
        if i < len(sets[7]):
            test.append(network.activate(sets[6][i]))  # test

    train_df = pd.DataFrame(train)
    validate_df = pd.DataFrame(validate)
    test_df = pd.DataFrame(test)

    min = 1.04964
    max = 1.5133

    training_setY = rescale(sets[1], min, max)
    validation_setY = rescale(sets[4], min, max)
    test_setY = rescale(sets[7], min, max)

    train_df = rescale(train_df, min, max)
    validate_df = rescale(validate_df, min, max)
    test_df = rescale(test_df, min, max)

    sets=[training_setY,validation_setY,test_setY,train_df,validate_df,test_df]
    error_train,error_val=evaluate_errors(sets,path)

    #save_network(network,path)

    #save_plots(training_setY, "Rzeczywiste", train_df, "Wytrenowane", sets[2], path, "train")
    #save_plots(test_setY, "Rzeczywiste", test_df, "Predykcja", sets[8], path, "test")
    #save_error_plot(path,"errors",train_errors)
    print(len(sets[2]),sets[2])


def save_error_plot(path,plot_name,list_of_errors):
    plt.plot(list_of_errors)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.savefig(path + '_' + plot_name+'.jpg', format='jpg', dpi=800)
    plt.close()

def save_plots(A, labelA, B, labelB, date, path, plot_name):
    #x_axis = [dt.datetime.strptime(d, '%d.%m.%Y').date() for d in date]
    #plt.plot(x_axis, A, label=labelA)
    #plt.plot(x_axis, B, label=labelB)
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.savefig(path + '_' + plot_name+'.jpg', format='jpg', dpi=1200)
    plt.close()

def save_network(network,path):
    networkk = open(path+"_network", 'wb')
    pickle.dump(network, networkk)
    networkk.close()