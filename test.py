from __future__ import print_function
from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from sys import stdout
from help_functions import *
import matplotlib.pyplot as plt

data = [1] * 3 + [2] * 3
data *= 3
print(data)

#ds = SequentialDataSet(1, 1)
#for sample, next_sample in zip(data, cycle(data[1:])):
    #ds.addSample(sample, next_sample)

set_path = "/home/adminz/repo/mgr/sets/"
save_results_path = "/home/adminz/results/RNN_close"
set_name = "close"

sets = generate_sets(set_path, set_name)
ds = SequentialDataSet(5,1)
for i in range(0, len(sets[1])):
    ds.addSample(sets[0][i], sets[1][i])

#ds.addSample([1,1],[0])
#ds.addSample([0,0],[0])
#ds.addSample([0,1],[1])
#ds.addSample([1,0],[1])

print(ds)

network = buildNetwork(5, 9, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

trainer = RPropMinusTrainer(network, dataset=ds)
train_errors = [] # save errors for plotting later
EPOCHS_PER_CYCLE = 5
CYCLES = 500
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in range(CYCLES):
    trainer.trainEpochs(EPOCHS_PER_CYCLE)
    train_errors.append(trainer.testOnData())
    epoch = (i+1) * EPOCHS_PER_CYCLE
    print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
    stdout.flush()

print()
print("final error =", train_errors[-1])

plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()
'''
for sample, target in ds.getSequenceIterator(0):
    print(sample)
    print("predicted next sample = %4.1f" % network.activate(sample))
    print("   actual next sample = %4.1f" % target)
    print()
'''

train = []
validate = []
test = []
for i in range(0, len(sets[1])):
    train.append(network.activate(sets[0][i])) #train
    if i < len(sets[4]):
        validate.append(network.activate(sets[3][i])) #validation
    if i < len(sets[7]):
        test.append(network.activate(sets[6][i])) #test

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

# test_setY = (test_setY * (max - min)) + min

# 0- train 1-val 2-test
#sets2 = [training_setY, validation_setY, test_setY, train_df, validate_df, test_df]
#error_train, error_val = evaluate_errors(sets2, path)

#save_network(network, path)
# '''
save_plots(training_setY, "Rzeczywiste", train_df,
           "Wytrenowane", sets[2], save_results_path, "train")
save_plots(test_setY, "Rzeczywiste", test_df,
           "Predykcja", sets[8], save_results_path, "test")
# '''

#print(error_train, error_val)