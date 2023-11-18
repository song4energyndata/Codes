# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

### Dataset 불러오기 (DNN은 여름평일/ 겨울평일/ 봄가을평일/ 휴일 각각에 대해 data 파일 별도로 구성함)

workday_springfall_train = np.loadtxt("I_train.txt")
workday_summer_train = np.loadtxt("II_train.txt")
workday_winter_train = np.loadtxt("III_train.txt")
holiday_train = np.loadtxt("IV_train.txt")

workday_springfall_test = np.loadtxt("I_test.txt")
workday_summer_test = np.loadtxt("II_test.txt")
workday_winter_test = np.loadtxt("III_test.txt")
holiday_test = np.loadtxt("IV_test.txt")

actualpower = np.loadtxt("actualusage.txt")

### DNN layer 설정 함수

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
        
### 훈련 및 예측 실행 함수 

def model_train_pred(data_train, data_test, modelname):
    response_train = data_train[:,1].reshape(-1,1)/1000 # [0,1] 구간즈음에 들어오게 scaling해야 함, 그렇지 않고 원본 쓰면 제대로 fit 안되고 대표값 1개만 반환함
    features_train = data_train[:,2:]
    hourindex_train = data_train[:,0].reshape(-1,1) 

    # response_test = data_test[:,1].reshape(-1,1)/1000
    features_test = data_test[:,2:]
    hourindex_test = data_test[:,0].reshape(-1,1) 

    model = model_dnn()

    checkpoint_cb = keras.callbacks.ModelCheckpoint("./"+modelname+".h5",monitor='loss',save_best_only=True) # monitor까지 해 줘야 파일이 저장됨
    model.compile(loss="mse",optimizer="adam") # 계측형 데이터 예측이므로 loss function은 mse
    history = model.fit(features_train,response_train,epochs=n_epochs,callbacks=[checkpoint_cb])

    y_fit = model.predict(features_train)
    y_pred = model.predict(features_test)

    return y_fit, y_pred, hourindex_train, hourindex_test

### 여름평일/ 겨울평일/ 봄가을평일/ 휴일 각각에 대해 모델 훈련 및 예측

[y_fit_workday_summer, y_pred_workday_summer, 
 hourindex_workday_summer_train, 
 hourindex_workday_summer_test] = model_train_pred(workday_summer_train, 
                                                   workday_summer_test, 
                                                   "dnn_workday_summer")

[y_fit_workday_winter, y_pred_workday_winter, 
 hourindex_workday_winter_train, 
 hourindex_workday_winter_test] = model_train_pred(workday_winter_train, 
                                                   workday_winter_test, 
                                                   "dnn_workday_winter")

[y_fit_workday_springfall, y_pred_workday_springfall, 
 hourindex_workday_springfall_train, 
 hourindex_workday_springfall_test] = model_train_pred(workday_springfall_train, 
                                                       workday_springfall_test, 
                                                       "dnn_workday_springfall")

[y_fit_holiday, y_pred_holiday, 
 hourindex_holiday_train, 
 hourindex_holiday_test] = model_train_pred(holiday_train, 
                                            holiday_test, 
                                            "dnn_holiday")

### 원래 시간 순서대로 데이터 재정렬

def aggregate(y_workday_summer, y_workday_winter, y_workday_springfall, y_holiday,
              hourindex_workday_summer, hourindex_workday_winter, hourindex_workday_springfall, hourindex_holiday):
    hourlypower = np.zeros((8760,))
    for k in range(y_workday_summer.shape[0]):
        hourlypower[int(hourindex_workday_summer[k][0]-1)] = y_workday_summer[k]
    for k in range(y_workday_winter.shape[0]):
        hourlypower[int(hourindex_workday_winter[k][0]-1)] = y_workday_winter[k]
    for k in range(y_workday_springfall.shape[0]):
        hourlypower[int(hourindex_workday_springfall[k][0]-1)] = y_workday_springfall[k]
    for k in range(y_holiday.shape[0]):
        hourlypower[int(hourindex_holiday[k][0]-1)] = y_holiday[k]
    return hourlypower*1000 # scale 원래대로

fittedpower = aggregate(y_fit_workday_summer, y_fit_workday_winter, y_fit_workday_springfall, y_fit_holiday,
                        hourindex_workday_summer_train, hourindex_workday_winter_train, hourindex_workday_springfall_train, hourindex_holiday_train)

predictedpower = aggregate(y_pred_workday_summer, y_pred_workday_winter, y_pred_workday_springfall, y_pred_holiday,
                        hourindex_workday_summer_test, hourindex_workday_winter_test, hourindex_workday_springfall_test, hourindex_holiday_test) 

### 모델 평가

def adjrsq(actual,estimate,k):
    bary = np.mean(actual)
    SST = np.sum((actual - bary)**2)
    SSR = np.sum((actual - estimate)**2)
    r = 1 - (SSR/(8760-k-1))/(SST/(8760-k))
    return r
    
adjRsq_train = adjrsq(actualpower[:,0],fittedpower,workday_summer_train[:,2:].shape[1]-2)
adjRsq_test = adjrsq(actualpower[:,1],predictedpower,workday_summer_test.shape[1]-2)

print(adjRsq_train)
print(adjRsq_test)


plt.plot(actualpower[:,0])
plt.plot(fittedpower)
plt.show()

plt.plot(actualpower[:,1])
plt.plot(predictedpower)
plt.show()