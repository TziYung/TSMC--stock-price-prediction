import pickle
import pandas
with open('TSMC.pickle','rb') as e:
    data=pickle.load(e)

dataf=pandas.DataFrame(data)

import numpy
train_data=[]
ans=[]
column=['change','change_rate','trade_volume','trade_value','P/E']
for a in range(1,len(dataf['opening'])-120):
    d=[]
    for b in range(7):
        d.append([])
    for b in range(1,10):
        l=[]
        for c in range(120):
            l.append(dataf.iloc[a+c,b])
        std=numpy.std(l)
        avg=numpy.average(l)
        for c in range(7):
            d[c].append((dataf.iloc[a+c,b]-avg)/std)
    d.reverse()
    train_data.append(d)
    sans=[]
    for b in range(2,4):
        l=[]
        for c in range(120):
            l.append(dataf.iloc[a+c,b])
        avg=numpy.average(l)
        std=numpy.std(l)
        sans.append((dataf.iloc[a-1,b]-avg)/std)
    ans.append(sans)


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
    keras.layers.LSTM(64,input_shape=train_data.shape[1:],return_sequences=True,activation='relu'),
    keras.layers.LSTM(64,return_sequences=True,activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32,activation='relu',return_sequences=False),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(2)
])
model.compile(loss='mean_squared_error',optimizer='adam')


model.fit(train_data,train_ans,epochs=100)
