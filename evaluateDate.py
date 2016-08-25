import pandas as pd


data = pd.read_csv("C:/Users/admin/Desktop/EURUSD2.csv", sep=';')
data2 = pd.read_csv("C:/Users/admin/Desktop/table.csv", sep=';')

ind=data['US500']
date=data['Date']
date2=data2['Date']

for i in range(0,data2['Close']):
    if(date2[i]==date[i]):
        date2[i]


print len(data2['Close'])