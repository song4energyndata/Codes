# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

data_train = np.loadtxt("RNN_train.txt")
data_test = np.loadtxt("RNN_test.txt")

n_node_firstlayer = 20
n_node_secondlayer = 10
n_epochs = 1000
optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)


response_train = data_train[:,0].reshape(1,-1) / 1000 # scaling
features_train = data_train[:,1:].reshape(1,-1,37) # reshape 해 줘야 rnn코드가 돌아감

response_test = data_test[:,0].reshape(1,-1) / 1000
features_test = data_test[:,1:].reshape(1,-1,37)



model_rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(n_node_firstlayer,return_sequences=True,input_shape=[None,37]),
    keras.layers.SimpleRNN(n_node_secondlayer,return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1)) # To make a sequence-to-sequence model, TimeDistributed should be used
    ])

checkpoint_cb_rnn = keras.callbacks.ModelCheckpoint("./rnn.h5",monitor='loss',save_best_only=True) # monitor까지 해 줘야 파일이 저장됨
model_rnn.compile(loss="mse",optimizer=optimizer)
history_rnn = model_rnn.fit(features_train,response_train,epochs=n_epochs,callbacks=[checkpoint_cb_rnn])

y_fit = (model_rnn.predict(features_train) * 1000).reshape(-1)
y_pred = (model_rnn.predict(features_test) * 1000).reshape(-1)



def adjrsq(actual,estimate,k):
    bary = np.mean(actual)
    SST = np.sum((actual - bary)**2)
    SSR = np.sum((actual - estimate)**2)
    r = 1 - (SSR/(8760-k-1))/(SST/(8760-k))
    return r
    
adjRsq_train = adjrsq(data_train[:,0],y_fit,features_train.shape[2]-2)
adjRsq_test = adjrsq(data_test[:,0],y_pred,features_test.shape[2]-2)

print(adjRsq_train)
print(adjRsq_test)



plt.plot(data_train[:,0])
plt.plot(y_fit)
plt.show()

plt.plot(data_test[:,0])
plt.plot(y_pred)
plt.show()

plt.plot((data_test[:,0] - y_fit))
