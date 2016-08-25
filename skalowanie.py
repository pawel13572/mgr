import pandas as pd
import csv
import os


data_all = pd.read_csv('EURUSD_full.csv',header=0,delimiter=';')
#print data_all.icol(2)
scaled = [[]]
for i in range(1,38):
    a=(data_all.icol(i)-data_all.icol(i).min())/(data_all.icol(i).max()-data_all.icol(i).min())
    scaled.append(a)

#csv_columns = ['Close0','Close1','Close2','Close3','Close4','Close5',
               #'MA5','MA10','MA20','MA60','MA120']
my_df = pd.DataFrame(data_all['Date'])

for j in range(1,38):
    my_df['i'+str(j)]=pd.DataFrame(scaled[j])


#print my_df

#my_df.to_csv('my_csv.csv', index=False, header=csv_columns)
my_df.to_csv('my_csv.csv', index=False)