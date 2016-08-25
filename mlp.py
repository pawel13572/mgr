from help_functions import *
import os

def full_evaluate_network(network_type,input,output,hidden,epochs,l_rate,network_name,folder_name,set_name):

    training_setX,training_setY,training_set_date =\
        generate_set("D:/PycharmProjects/Forex/sets/" + set_name + "_training_set.csv",
         "D:/PycharmProjects/Forex/sets/" + set_name + "_EURUSD.csv", (0, 1))

    validation_setX,validation_setY,val_date=\
        generate_set("D:/PycharmProjects/Forex/sets/" + set_name + "_validation_set.csv",
    "D:/PycharmProjects/Forex/sets/" + set_name + "_EURUSD.csv", (0, 1))

    test_setX,test_setY,test_set_date =\
        generate_set("D:/PycharmProjects/Forex/sets/" + set_name + "_test_set.csv",
                     "D:/PycharmProjects/Forex/sets/" + set_name + "_EURUSD.csv", (0, 1))

    if(network_type<2):
         network = generate_network(input, hidden, output, network_type)
    else:
         network = generate_linear_network(input,output)

    network_name= network_name + '_' + str(input) + '_' + str(hidden) + '_' + str(output) + '_' + str(epochs) + 'e'
    folder_name2=network_name

    path = folder_name + '/' + folder_name2 + '/' + network_name
    full_path='D:/PycharmProjects/Forex/' + folder_name + '/' + folder_name2

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    ds = SupervisedDataSet(input, output)
    for i in range(0,len(training_setY)):
        ds.addSample(training_setX[i],training_setY[i])

    trainer = BackpropTrainer(network, ds, learningrate=l_rate, verbose=False)
    trainer.train()
    errors=[]
    errors.append(trainer.trainEpochs(epochs=1))
    print errors

    train=[]

    '''
    for m in range(0,1000):
        trainer.trainEpochs(1)
        if (m%10==0):
            l_rate=l_rate*0.9
            print l_rate
            trainer = BackpropTrainer(network, ds, learningrate=l_rate, verbose=True)
            for i in range(0,len(training_setY)):
                train.append(network.activate(training_setX[i]))
            plt.plot(training_setY)
            plt.plot(train)
            plt.show(block=False)
            plt.pause(0.0000000001)

            del train[:]
            plt.clf()
        '''


    validate=[]
    test=[]
    for i in range(0,len(training_setY)):
        train.append(network.activate(training_setX[i]))
        if i < len(validation_setY):
            validate.append(network.activate(validation_setX[i]))
        if i < len(test_setY):
            test.append(network.activate(test_setX[i]))

    train_df = pd.DataFrame(train)
    validate_df = pd.DataFrame(validate)
    test_df = pd.DataFrame(test)

    min = 1.04964
    max = 1.5133

    training_setY=rescale(training_setY,min,max)
    validation_setY=rescale(validation_setY,min,max)
    test_setY= rescale(test_setY,min,max)

    train_df=rescale(train_df,min,max)
    validate_df=rescale(validate_df,min,max)
    test_df= rescale(test_df,min,max)

    #test_setY = (test_setY * (max - min)) + min

    #0- train 1-val 2-test
    sets=[training_setY,validation_setY,test_setY,train_df,validate_df,test_df]
    error_train,error_val=evaluate_errors(sets,path)

    save_network(network,path)
    #'''
    save_plots(training_setY, "Rzeczywiste", train_df,
                              "Wytrenowane", training_set_date, path,"train")
    save_plots(test_setY, "Rzeczywiste", test_df,
                              "Predykcja", test_set_date, path,"test")
                              #'''

    return error_train,error_val