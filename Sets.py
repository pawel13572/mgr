import pandas as pd
from random import randint

path="C:/Users/admin/PycharmProjects/Forex/sets/"

data_all = pd.read_csv(path + 'select_EURUSD.csv')
rand = pd.read_csv(path + 'random_list.csv')
numbers= rand['numbers']
save_name="select"



val_train_set=1928-193

test_set=data_all.tail(193)
set_set=data_all[0:val_train_set]
#rand_list=[]
#while len(rand_list)<385:
#    x=randint(0,val_train_set)
#    if (x in rand_list) == False:
#        rand_list.append(x)
#
#my_df = pd.DataFrame(rand_list)
#my_df.to_csv('random_list.csv',index=False)

val_set=[]
train_set=[]
for i in range(0,val_train_set):
    c=0
    for j in range(385):
        if numbers[j]==i:
            c=c+1
            val_set.append(data_all.loc[i])
    if (c==0):
        train_set.append(data_all.loc[i])

val_df = pd.DataFrame(val_set)
val_df.to_csv(path + save_name+'_validation_set.csv',index=False)

train_df = pd.DataFrame(train_set)
train_df.to_csv(path + save_name+'_training_set.csv',index=False)

test_set.to_csv(path + save_name+'_test_set.csv',index=False)




