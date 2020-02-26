import pickle
import pandas
with open('TSMC.pickle','rb') as e:
    data=pickle.load(e)
dataf=pandas.DataFrame(data)

import numpy
train_data=[]
ans=[]
column=['change','change_rate','trade_volume','trade_value','P/E']
for a in range(1,len(dataf['opening'])-30):
    d=[]
    ans.append(list(dataf.iloc[a-1,[2,3]])) # the highest and lowest price of the day
    for b in range(7):
        d.append(list(dataf.iloc[a+b,[1,2,3,4]])) # the open,high,low,close price of the day
    for b in range(5,10):
        l=[]
        for c in range(30):
            l.append(dataf.iloc[a+c,b])
        std=numpy.std(l)
        avg=0
        for c in range(30):
            avg+=l[c]# std and avg of past 30 days
        avg=avg/30
        for c in range(7):
            d[c].append((dataf.iloc[a+c,b]-avg)/std)
    d.reverse()
    train_data.append(d)

import numpy
test_data=train_data[:50]
train_data=train_data[50:]
test_ans=ans[:50]
train_ans=ans[50:]
train_data=numpy.array(train_data)
test_data=numpy.array(test_data)
test_ans=numpy.array(test_ans)
train_ans=numpy.array(train_ans)

from tensorflow import keras
model=keras.Sequential([
    keras.layers.LSTM(140,input_shape=train_data.shape[1:],return_sequences=False,activation='relu'),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(2)
])
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

model.fit(train_data,train_ans,epochs=100)
