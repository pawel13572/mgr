from mlp import *
import pandas as pd
from datetime import datetime
import numpy as np
# 0- MLP , 1- rec, 2-linear

#close
#full_evaluate_network(0,5,1,22,10000,0.1,"closeNetwork","MLP_close","close")

#ma
#full_evaluate_network(0,6,1,10,10000,0.1,"maNetwork","MLP_ma","ma")

#rec close
#1- rec
#full_evaluate_network(1,5,1,30,10000,0.1,"closeRecNetwork","Rec_close","close")

#rec ma
#1- rec
#full_evaluate_network(1,6,1,73,5,0.9,"maRecNetwork","Rec_ma","ma")

#2- linear
#full_evaluate_network(3, 4, 1, 2, 100, 0.03, "selectSetNetwork", "linear", "select")

#full_evaluate_network(0,5,1,31,20000,0.07,"closeNetwork","MLP_close","close")

#fileObject = open('C:/Users/admin/PycharmProjects/Forex/full/network','r')
#net = pickle.load(fileObject)
'''
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
               '''

def pesos_conexiones(n):

    contribution=[]
    x=[]
    for m in range(0,36):
        contribution.append(0)
        x.append(m)

    a=0
    for mod in n.modules:
        for conn in n.connections[mod]:
            if a==0 or a==37:
                print conn
                a=a+1
                for cc in range(len(conn.params)):
                    for j in range(0,36):
                        for i in range(0, 9):
                            if conn.whichBuffers(cc)==(j,i):
                                #print conn.whichBuffers(cc), conn.params[cc]
                                contribution[j]=contribution[j]+abs(conn.params[cc])
            else:
                print conn
                a=a+1
                for cc in range(len(conn.params)):
                    for j in range(0,36):
                        for i in range(0, 9):
                            if conn.whichBuffers(cc)==(j,i):
                                print conn.whichBuffers(cc), conn.params[cc]
                                #contribution[j]=contribution[j]+abs(conn.params[cc])
    #print contribution
    #array=np.asarray(contribution)
    #plt.bar(contribution)
    #print range(contribution)
    #x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    plt.bar(x,contribution, color="black")

    #plt.plot(contribution)
    plt.show()


#pesos_conexiones(net)


'''
for c in [connection for connections in net.connections.values() for connection in connections]:
    print("{} -> {} => {}".format(c.inmod.name, c.outmod.name, c.params))
    print("org {}", np.reshape(c.params, (c.outdim, c.indim)))
'''

'''
training_setX, training_setY, training_set_date = \
    generate_set("C:/Users/admin/PycharmProjects/Forex/sets/full_EURUSD.csv",
                 "C:/Users/admin/PycharmProjects/Forex/sets/full_EURUSD.csv", (0, 1))
net = generate_network(36, 9, 1, 0)

full_path = 'C:/Users/admin/PycharmProjects/Forex/full'

if not os.path.exists(full_path):
    os.makedirs(full_path)

ds = SupervisedDataSet(36, 1)
for i in range(0, len(training_setY)):
    ds.addSample(training_setX[i], training_setY[i])

trainer = BackpropTrainer(net, ds, learningrate=0.1, verbose=True)
trainer.train()
trainer.trainEpochs(epochs=1000)

save_network(net,'full/network')

train=[]

for i in range(0, len(training_setY)):
    train.append(net.activate(training_setX[i]))

plt.plot(train)
plt.plot(training_setY)
plt.savefig('C:/Users/admin/PycharmProjects/Forex/full/full_wykres.jpg', format='jpg', dpi=1200)
plt.close()

#pesos_conexiones(net)
'''

#text_close_MLP = open('C:/Users/admin/PycharmProjects/Forex/MLP_close/close_mlp_errors.txt', "w")

def search_network():

    closeMlpTrainTab=[]
    closeMlpValTab=[]
    closeMlpError=[]

    empty=[]

    maMlpTrainTab=[]
    maMlpValTab=[]
    maMlpError=[]

    closeRecTrainTab=[]
    closeRecValTab=[]
    closeRecError=[]

    maRecTrainTab=[]
    maRecValTab=[]
    maRecError=[]

    Layers=[]

    a = datetime.now()
    iterations=200
    for i in range(2,100):
        Layers.append(i)

        error_train,error_val=full_evaluate_network(0,5,1,i,iterations,0.1,"closeNetwork","MLP_close","close")
        closeMlpTrainTab.append(error_train)
        closeMlpValTab.append(error_val)
        closeMlpError.append(math.pow(error_train-error_val,2)/2)

        error_train, error_val = full_evaluate_network(0,6,1,i,iterations,0.1,"maNetwork","MLP_ma","ma")
        maMlpTrainTab.append(error_train)
        maMlpValTab.append(error_val)
        maMlpError.append(math.pow(error_train-error_val,2)/2)

        error_train, error_val = full_evaluate_network(1,5,1,i,iterations,0.003,"closeRecNetwork","Rec_close","close")
        closeRecTrainTab.append(error_train)
        closeRecValTab.append(error_val)
        closeRecError.append(math.pow(error_train-error_val,2)/2)

        error_train, error_val = full_evaluate_network(1,6,1,i,iterations,0.003,"maRecNetwork","Rec_ma","ma")
        maRecTrainTab.append(error_train)
        maRecValTab.append(error_val)
        maRecError.append(math.pow(error_train-error_val,2)/2)

        b = datetime.now()

        c=b-a

        print "Layers: ", i, "  Time: ", c


    table = pd.DataFrame(Layers)

    table["close_mlp_Train"]=pd.DataFrame(closeMlpTrainTab)
    table["close_mlp_Val"]=pd.DataFrame(closeMlpValTab)
    table["close_mlp_Error"]=pd.DataFrame(closeMlpError)

    table["ma_mlp_Train"]=pd.DataFrame(maMlpTrainTab)
    table["ma_mlp_Val"]=pd.DataFrame(maMlpValTab)
    table["ma_mlp_Error"]=pd.DataFrame(maMlpError)

    table["close_rec_Train"]=pd.DataFrame(closeRecTrainTab)
    table["close_rec_Val"]=pd.DataFrame(closeRecValTab)
    table["close_rec_Error"]=pd.DataFrame(closeRecError)

    table["ma_rec_Train"]=pd.DataFrame(maRecTrainTab)
    table["ma_rec_Val"]=pd.DataFrame(maRecValTab)
    table["ma_rec_Error"]=pd.DataFrame(maRecError)

    writer = pd.ExcelWriter('D:/PycharmProjects/Forex/errors.xlsx')
    table.to_excel(writer)
    writer.save()

#search_network()


