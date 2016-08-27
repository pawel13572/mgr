from __future__ import print_function
from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer,BackpropTrainer
from sys import stdout
from help_functions import *
import matplotlib.pyplot as plt


set_path = "/home/adminz/repo/mgr/sets/"
save_results_path = "/home/adminz/results/RNN_close/"
set_name = "close"

sets = generate_sets(set_path, set_name)
ds = generate_sequential_data_set(sets[0],sets[1])
n = generate_rnn_network(input_neurons=len(sets[0][0]),hidden_neurons=9,output_neurons=1)
trained_network,train_error=train_network(network=n,data_set=ds,epochs=10)
generate_plots_and_errors(save_results_path=save_results_path,set_name=set_name,
                          network=trained_network,train_errors=train_error, sets=sets)

