from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer
from pybrain.structure import SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import pandas as pd
import matplotlib.pyplot as plt

data_all = pd.read_csv('EURUSD_Scaled.csv',header=0, usecols=[1,2,3,4,5,6],names=['Close0', 'Close1', 'Close2', 'Close3', 'Close4', 'Close5'],delimiter=';')

close1= data_all['Close1']
close2= data_all['Close2']
close3= data_all['Close3']
close4= data_all['Close4']
close5= data_all['Close5']

close0= data_all['Close0']

n = RecurrentNetwork()

n.addInputModule(LinearLayer(5, name='in'))
n.addModule(SigmoidLayer(3, name='hidden'))
n.addOutputModule(SigmoidLayer(1, name='out'))

n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))

n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))

n.sortModules()

ds = SupervisedDataSet(5, 1)

for i in range(0,len(close0)):
    ds.addSample((close1[i],close2[i],close3[i],close4[i],close5[i]),(close0[i]))

trainer = BackpropTrainer(n, ds,learningrate=0.7,verbose=True)
trainer.trainEpochs(epochs=10)
trainer.train()

train=[]
for i in range(0,len(close0)):
    train.append(n.activate((close1[i],close2[i],close3[i],close4[i],close5[i])))


plt.plot(close0)
plt.plot(train)
plt.show()


