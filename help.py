import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pybrain.tools.shortcuts import buildNetwork

#data = pd.read_csv("ma_EURUSD.csv", header=0, usecols=[1,7], delimiter=';')
#plt.plot(data['Close0'],label="Szereg czasowy")
#plt.plot(data['MA120'],label="Srednia kroczaca")
#plt.legend()
#plt.show()

fileObject = open('/home/adminz/results/validation/mlp_close2/_network','rb')
net = pickle.load(fileObject)
for mod in net.modules:
    print("Module:", mod.name)
    if mod.paramdim > 0:
        print("--parameters:", mod.params)
    for conn in net.connections[mod]:
        print("-connection to", conn.outmod.name)
        if conn.paramdim > 0:
             print("- parameters", conn.params)
    if hasattr(net, "recurrentConns"):
        print("Recurrent connections")
        for conn in net.recurrentConns:
            print("-", conn.inmod.name, " to", conn.outmod.name)
            if conn.paramdim > 0:
                print("- parameters", conn.params)

fileObject = open('/home/adminz/results/validation/mlp_ma2/_network','rb')
net = pickle.load(fileObject)
for mod in net.modules:
    print("Module:", mod.name)
    if mod.paramdim > 0:
        print("--parameters:", mod.params)
    for conn in net.connections[mod]:
        print("-connection to", conn.outmod.name)
        if conn.paramdim > 0:
             print("- parameters", conn.params)
    if hasattr(net, "recurrentConns"):
        print("Recurrent connections")
        for conn in net.recurrentConns:
            print("-", conn.inmod.name, " to", conn.outmod.name)
            if conn.paramdim > 0:
                print("- parameters", conn.params)