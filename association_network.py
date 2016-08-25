from pybrain.structure import LinearLayer, SigmoidLayer,FullConnection,FeedForwardNetwork,LinearConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from mlp import *


n = FeedForwardNetwork()

inLayer = LinearLayer(36,name='inp')
hiddenLayer1 = SigmoidLayer(23,name='hid1')
hiddenLayer2 = SigmoidLayer(10,name='hid2')
hiddenLayer3 = SigmoidLayer(23,name='hid3')
outLayer = SigmoidLayer(36,name='out')

n.addInputModule(inLayer)
n.addModule(hiddenLayer1)
n.addModule(hiddenLayer2)
n.addModule(hiddenLayer3)
n.addOutputModule(outLayer)

in_to_hidden1 = FullConnection(inLayer, hiddenLayer1,name='c1')
hidden1_to_hidden2 = FullConnection(hiddenLayer1, hiddenLayer2,name='c2')
hidden2_to_hidden3 = FullConnection(hiddenLayer2, hiddenLayer3,name='c3')
hidden3_to_out = FullConnection(hiddenLayer3, outLayer,name='c4')

n.addConnection(in_to_hidden1)
n.addConnection(hidden1_to_hidden2)
n.addConnection(hidden2_to_hidden3)
n.addConnection(hidden3_to_out)

n.sortModules()

training_setX, training_setY, training_set_date = \
    generate_set("C:/Users/admin/PycharmProjects/Forex/sets/full_EURUSD.csv",
                 "C:/Users/admin/PycharmProjects/Forex/sets/full_EURUSD.csv", (0, 1))

ds = SupervisedDataSet(36, 36)
for i in range(0, len(training_setY)):
    ds.addSample(training_setX[i], training_setX[i])

trainer = BackpropTrainer(n, ds, learningrate=0.1, verbose=True)
trainer.train()
trainer.trainEpochs(epochs=10000)

save_network(n,'as_network/network')