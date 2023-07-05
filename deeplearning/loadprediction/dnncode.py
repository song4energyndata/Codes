# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

workday_springfall_train = np.loadtxt("I_train.txt")
workday_summer_train = np.loadtxt("II_train.txt")
workday_winter_train = np.loadtxt("III_train.txt")
holiday_train = np.loadtxt("IV_train.txt")

workday_springfall_test = np.loadtxt("I_test.txt")
workday_summer_test = np.loadtxt("II_test.txt")
workday_winter_test = np.loadtxt("III_test.txt")
holiday_test = np.loadtxt("IV_test.txt")

actualpower = np.loadtxt("actualusage.txt")

n_node_firstlayer = 20
n_node_secondlayer = 10
n_epochs = 1000

def model_dnn():
    model = keras.models.Sequential([
            keras.layers.Dense(n_node_firstlayer,activation="sigmoid"),
            keras.layers.Dense(n_node_secondlayer,activation="sigmoid"),
            keras.layers.Dense(1,activation="linear") # 계측형 데이터 예측이므로 마지막 층은 node 1개, activation은 linear
            ])
    return model
        


response_workday_summer_train = workday_summer_train[:,1].reshape(-1,1)/1000 # [0,1] 구간즈음에 들어오게 scaling해야 함, 그렇지 않고 원본 쓰면 제대로 fit 안되고 대표값 1개만 반환함
features_workday_summer_train = workday_summer_train[:,2:]
hourindex_workday_summer_train = workday_summer_train[:,0].reshape(-1,1) 

response_workday_summer_test = workday_summer_test[:,1].reshape(-1,1)/1000
features_workday_summer_test = workday_summer_test[:,2:]
hourindex_workday_summer_test = workday_summer_test[:,0].reshape(-1,1) 

model_workday_summer = model_dnn()

checkpoint_cb = keras.callbacks.ModelCheckpoint("./dnn_workday_summer.h5",monitor='loss',save_best_only=True) # monitor까지 해 줘야 파일이 저장됨
model_workday_summer.compile(loss="mse",optimizer="adam") # 계측형 데이터 예측이므로 loss function은 mse
history = model_workday_summer.fit(features_workday_summer_train,response_workday_summer_train,epochs=n_epochs,callbacks=[checkpoint_cb])

y_fit_workday_summer = model_workday_summer.predict(features_workday_summer_train)
y_pred_workday_summer = model_workday_summer.predict(features_workday_summer_test)



response_workday_winter_train = workday_winter_train[:,1].reshape(-1,1)/1000 # [0,1] 구간즈음에 들어오게 scaling해야 함, 그렇지 않고 원본 쓰면 제대로 fit 안되고 대표값 1개만 반환함
features_workday_winter_train = workday_winter_train[:,2:]
hourindex_workday_winter_train = workday_winter_train[:,0].reshape(-1,1) 

response_workday_winter_test = workday_winter_test[:,1].reshape(-1,1)/1000
features_workday_winter_test = workday_winter_test[:,2:]
hourindex_workday_winter_test = workday_winter_test[:,0].reshape(-1,1) 

model_workday_winter = model_dnn()

checkpoint_cb = keras.callbacks.ModelCheckpoint("./dnn_workday_winter.h5",monitor='loss',save_best_only=True) # monitor까지 해 줘야 파일이 저장됨
model_workday_winter.compile(loss="mse",optimizer="adam") # 계측형 데이터 예측이므로 loss function은 mse
history = model_workday_winter.fit(features_workday_winter_train,response_workday_winter_train,epochs=n_epochs,callbacks=[checkpoint_cb])

y_fit_workday_winter = model_workday_winter.predict(features_workday_winter_train)
y_pred_workday_winter = model_workday_winter.predict(features_workday_winter_test)



response_workday_springfall_train = workday_springfall_train[:,1].reshape(-1,1)/1000 # [0,1] 구간즈음에 들어오게 scaling해야 함, 그렇지 않고 원본 쓰면 제대로 fit 안되고 대표값 1개만 반환함
features_workday_springfall_train = workday_springfall_train[:,2:]
hourindex_workday_springfall_train = workday_springfall_train[:,0].reshape(-1,1) 

response_workday_springfall_test = workday_springfall_test[:,1].reshape(-1,1)/1000
features_workday_springfall_test = workday_springfall_test[:,2:]
hourindex_workday_springfall_test = workday_springfall_test[:,0].reshape(-1,1) 

model_workday_springfall = model_dnn()

checkpoint_cb = keras.callbacks.ModelCheckpoint("./dnn_workday_springfall.h5",monitor='loss',save_best_only=True) # monitor까지 해 줘야 파일이 저장됨
model_workday_springfall.compile(loss="mse",optimizer="adam") # 계측형 데이터 예측이므로 loss function은 mse
history = model_workday_springfall.fit(features_workday_springfall_train,response_workday_springfall_train,epochs=n_epochs,callbacks=[checkpoint_cb])

y_fit_workday_springfall = model_workday_springfall.predict(features_workday_springfall_train)
y_pred_workday_springfall = model_workday_springfall.predict(features_workday_springfall_test)



response_holiday_train = holiday_train[:,1].reshape(-1,1)/1000 # [0,1] 구간즈음에 들어오게 scaling해야 함, 그렇지 않고 원본 쓰면 제대로 fit 안되고 대표값 1개만 반환함
features_holiday_train = holiday_train[:,2:]
hourindex_holiday_train = holiday_train[:,0].reshape(-1,1) 

response_holiday_test = holiday_test[:,1].reshape(-1,1)/1000
features_holiday_test = holiday_test[:,2:]
hourindex_holiday_test = holiday_test[:,0].reshape(-1,1) 

model_holiday = model_dnn()

checkpoint_cb = keras.callbacks.ModelCheckpoint("./dnn_holiday.h5",monitor='loss',save_best_only=True) # monitor까지 해 줘야 파일이 저장됨
model_holiday.compile(loss="mse",optimizer="adam") # 계측형 데이터 예측이므로 loss function은 mse
history = model_holiday.fit(features_holiday_train,response_holiday_train,epochs=n_epochs,callbacks=[checkpoint_cb])

y_fit_holiday = model_holiday.predict(features_holiday_train)
y_pred_holiday = model_holiday.predict(features_holiday_test)



fittedpower = np.zeros((8760,))
for k in range(y_fit_workday_summer.shape[0]):
    fittedpower[int(hourindex_workday_summer_train[k][0]-1)] = y_fit_workday_summer[k]
for k in range(y_fit_workday_winter.shape[0]):
    fittedpower[int(hourindex_workday_winter_train[k][0]-1)] = y_fit_workday_winter[k]
for k in range(y_fit_workday_springfall.shape[0]):
    fittedpower[int(hourindex_workday_springfall_train[k][0]-1)] = y_fit_workday_springfall[k]
for k in range(y_fit_holiday.shape[0]):
    fittedpower[int(hourindex_holiday_train[k][0]-1)] = y_fit_holiday[k]
    
predictedpower = np.zeros((8760,))
for k in range(y_pred_workday_summer.shape[0]):
    predictedpower[int(hourindex_workday_summer_test[k][0]-1)] = y_pred_workday_summer[k]
for k in range(y_pred_workday_winter.shape[0]):
    predictedpower[int(hourindex_workday_winter_test[k][0]-1)] = y_pred_workday_winter[k]
for k in range(y_pred_workday_springfall.shape[0]):
    predictedpower[int(hourindex_workday_springfall_test[k][0]-1)] = y_pred_workday_springfall[k]
for k in range(y_pred_holiday.shape[0]):
    predictedpower[int(hourindex_holiday_test[k][0]-1)] = y_pred_holiday[k]    
    
fittedpower = fittedpower*1000   
predictedpower = predictedpower*1000   



def adjrsq(actual,estimate,k):
    bary = np.mean(actual)
    SST = np.sum((actual - bary)**2)
    SSR = np.sum((actual - estimate)**2)
    r = 1 - (SSR/(8760-k-1))/(SST/(8760-k))
    return r
    
adjRsq_train = adjrsq(actualpower[:,0],fittedpower,features_workday_summer_train.shape[1]-2)
adjRsq_test = adjrsq(actualpower[:,1],predictedpower,features_workday_summer_test.shape[1]-2)

print(adjRsq_train)
print(adjRsq_test)


plt.plot(actualpower[:,0])
plt.plot(fittedpower)
plt.show()

plt.plot(actualpower[:,1])
plt.plot(predictedpower)
plt.show()