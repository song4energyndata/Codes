# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow import keras
import time

### hyperparameters

optimizer = keras.optimizers.Adam(lr=0.005) # learning rate 너무 높으면 발산할 수 있음에 주의
batch_size = 32
discount_factor = 0.98
period_step_fortrain = 24
rewardscalefactor = 1/6
buffer_size = 20000



### microgrid system data

# 시간별 데이터 출처: https://github.com/VinF/deer/tree/master/examples/MG_two_storages/data
PV_prod_train = np.load('BelgiumPV_prod_train.npy') # [0,1] 구간 내로 scaled된 데이터임, unscaling은 play_one_step 함수 내에서 이루어짐
PV_prod_test  = np.load('BelgiumPV_prod_test.npy') 

load_train = np.load('example_nondeterminist_cons_train.npy') # [0,1] 구간 내로 scaled된 데이터임, unscaling은 play_one_step 함수 내에서 이루어짐
load_test  = np.load('example_nondeterminist_cons_test.npy')

prate_h2 = 1.1 # 수전/송전 상한
eff_h2 = 0.65 # 수소-전기 변환효율

capa_batt = 15 # 배터리 용량 (겉보기용량 말고 SOC 상하한 고려한 실용량이라 가정)
eff_batt = 0.9 # 배터리 충방전 효율
initialenergy_batt = 0.0 # epoch의 시작에서 배터리 내 에너지량 (최대저장가능량 대비 상대비율, Qval NN에는 이 상대비율이 입력되며, play_one_step 함수 내에서의 에너지시스템 밸런스 수식에서는 실제 에너지값으로 변환됨

price_h2 = 0.1
cost_loss = 2

load_peak = 2
pv_peak = 12

inputlen_load = 24
inputlen_pv = 24



### Neural net configuration

n_outputs = 3
input_load = keras.layers.Input(shape=(inputlen_load,1)) # 과거 24시간의 부하, shape의 두번째 숫자는 채널 수 (컬러사진의 RGB 등), 채널 수를 정의해야 Conv1D가 작동함
input_pv = keras.layers.Input(shape=(inputlen_pv,1)) # 과거 24시간의 태양광발전량
hidden_conv_load = keras.layers.Conv1D(16, 2, activation='relu')(input_load)
hidden_conv_pv = keras.layers.Conv1D(16, 2, activation='relu')(input_pv)
concat_conv = keras.layers.concatenate([hidden_conv_load,hidden_conv_pv],axis=2)
hidden_conv_concat = keras.layers.Conv1D(16,2,activation='relu')(concat_conv)
flatten_conv = tf.keras.layers.Flatten()(hidden_conv_concat) # Dense층 직전에 Conv층을 Flatten해야 함
input_others = keras.layers.Input(shape=(1)) # 직전 시간의 배터리 내 에너지량
concat_all = keras.layers.concatenate([flatten_conv,input_others]) 
hidden_dense_1 = keras.layers.Dense(50,activation="relu")(concat_all)
hidden_dense_2 = keras.layers.Dense(20,activation="relu")(hidden_dense_1)
output = keras.layers.Dense(n_outputs)(hidden_dense_2)

model = keras.Model(inputs=[input_load,input_pv,input_others], outputs=[output])


### Policy에 따른 state 별 action 결정 함수

def e_greedy_policy(state,epsilon=0): # epsilon-greedy policy
    if np.random.rand() < epsilon: # epsilon의 확률로 exploration
        return np.random.randint(3) # action을 랜덤하게 선택
    else: # exploitation
        input_load = state[0:inputlen_load].reshape(-1,inputlen_load,1) # Conv1D의 input이므로 채널 수 1 명시
        input_pv = state[inputlen_load:(inputlen_load+inputlen_pv)].reshape(-1,inputlen_pv,1)
        input_others =state[(inputlen_load+inputlen_pv)].reshape(-1,1) # Dense층의 input이므로 채널 수는 필요 없음
        Q_values = model((input_load,input_pv,input_others)) # 각 action 별 Q-value 도출
        return np.argmax(Q_values[0]) # 가장 큰 Q-value에 대응하는 action 선택      
        # 주의: for loop 안에서 DNN에 input을 입력해 output 계산 시 model()로 해야지, 
        # model.predict()로 하면 안 됨! 메모리 누수가 발생함 
        # (model.predict는 대량의 input data를 'model.predict를 한 번만 호출해서' 처리하는 데 특화됨, https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict 참고)


### replay buffer 정의

from collections import deque
replay_buffer = deque(maxlen=buffer_size) 

def sample_experiences(batch_size): # batch_size만큼의 경험들의 state들, action들, reward들, nextstate들의 리스트들을 반환
    indices = np.random.randint(len(replay_buffer), size=batch_size) # replay buffer 내 경험들 중 랜덤하게 batch_size만큼의 경험들을 지정 (인덱스 불러옴)
    batch = [replay_buffer[index] for index in indices] # 위에서 불러온 인덱스를 이용해 batch_size 만큼의 경험들을 batch 리스트에 담음
    states, actions, rewards, next_states = [
        np.array([experience[field_index] for experience in batch]) # 하나의 array 안에 batch 내 각 경험별 값들이 들어감 
        for field_index in range(4)] # 총 4개의 array를 반환, 각각은 (batch 내 sample경험들의) state들, action들, reward들, nextstate들의 리스트임
    return states, actions, rewards, next_states



### 1 time step에 대해 action 수행

def play_one_step(profile_load,profile_pv,hour,energy_batt,epsilon,training=False): # 마이크로그리드 시스템의 시간별 operation 모델링
        
    state = np.concatenate( (profile_load[hour-inputlen_load:hour], profile_pv[hour-inputlen_pv:hour], np.array([energy_batt])) )
    
    action = e_greedy_policy(state,epsilon) # epsilon-greedy policy에 따라 action 도출
    p_h2_send = 0
    p_h2_receive = 0
    if action == 0: # 수소로 생산된 전기 수전
        p_h2_receive = prate_h2
    elif action == 1: # 수소 생산용 전기 송전
        p_h2_send = prate_h2
       

    p_load = profile_load[hour]*load_peak # 현재 시간의 부하 (action 결정 기준이 아님!), 입력자료가 [0,1]로 scaled된 걸 unscaling
    p_pv = profile_pv[hour]*pv_peak # 현재 시간의 발전량 (action 결정 기준이 아님!), 입력자료가 [0,1]로 scaled된 걸 unscaling  
    energy_batt = energy_batt*capa_batt # 입력자료가 [0,1]로 scaled된 걸 unscaling
    
    #p_curtail = 0
    p_loss = 0
    
    p_net_beforebatt = p_pv - p_load + p_h2_receive - p_h2_send # 태양광 생산, 부하 충족, H2 보내거나 받은 후 수용가 입장에서 전기에너지가 남으면 양수, 부족하면 음수
    
    if p_net_beforebatt >= 0: # 전기에너지가 남으므로 배터리에 저장하고, 만약 배터리도 꽉 찬다면 curtail함
        if capa_batt >= energy_batt + p_net_beforebatt*eff_batt: # 배터리에 잔여 충전 가능한 에너지가 위에서 남은 에너지(에서 변환손실 제한 에너지) 이상일 경우
            energy_batt_after = energy_batt + p_net_beforebatt*eff_batt
        else: # 남은 에너지를 배터리에 다 충전하면 꽉 차고도 남아서, curtail해야 함
            energy_batt_after = capa_batt
            #p_curtail = (energy_batt + p_net_beforebatt*eff_batt - capa_batt)/eff_batt # 괄호 안은 배터리 내부 기준이며, 수용가 모선 기준 양을 구하려면 eff_batt로 나눠줘야 함
    else: # 전기에너지가 부족하므로 배터리 에너지를 써야 함, 배터리 에너지로도 부족하면 loss임
        if energy_batt*eff_batt >= -p_net_beforebatt: # 배터리 에너지로 충당 가능한 경우
            energy_batt_after = energy_batt + p_net_beforebatt/eff_batt # p_net_beforebatt가 음수이므로 마이너스값이 부족한 에너지'량'이고 그걸 '빼'므로 결과적으로 플러스
        else: # 부족해 loss 발생
            energy_batt_after = 0
            p_loss = -p_net_beforebatt - energy_batt*eff_batt           
    
    reward = price_h2*p_h2_send*eff_h2 - price_h2*p_h2_receive/eff_h2 - cost_loss*p_loss
    energy_batt_after = energy_batt_after/capa_batt # 배터리 내 저장량을 [0,1] 범위로 scaling함
    
    if training == True:
        next_state = np.concatenate( (profile_load[hour-(inputlen_load-1):hour+1], profile_pv[hour-(inputlen_pv-1):hour+1], np.array([energy_batt_after])) )         
        replay_buffer.append((state, action, reward*rewardscalefactor, next_state)) # replay buffer에는 scaled reward를 넣으며, tuple을 append함에 주의
    return energy_batt_after, reward, action # scaled 배터리 내 저장량, unscaled reward, action index 반환


### 심층신경망 가중치 업데이트 수행

def training_step(batch_size): 

    states, actions, rewards, next_states = sample_experiences(batch_size)
        
    input_load = states[:,0:inputlen_load].reshape(-1,inputlen_load,1) 
    input_pv = states[:,inputlen_load:(inputlen_load+inputlen_pv)].reshape(-1,inputlen_pv,1)
    input_others = states[:,(inputlen_load+inputlen_pv)].reshape(-1,1)
    
    input_load_next = next_states[:,0:inputlen_load].reshape(-1,inputlen_load,1)
    input_pv_next = next_states[:,inputlen_load:(inputlen_load+inputlen_pv)].reshape(-1,inputlen_pv,1)
    input_others_next = next_states[:,(inputlen_load+inputlen_pv)].reshape(-1,1)
      
    next_Q_values = model((input_load_next,input_pv_next,input_others_next)) # Q-learning을 위해, 'next'state에서의 각 action 별 Q-value 추정치들 반환
    max_next_Q_values = np.max(next_Q_values, axis=1) # 각 경험별로 nextstate에 대한 Q-value가 더 높은 행동의 Q-value 사용
    target_Q_values = (rewards + discount_factor * max_next_Q_values) # Q-value를 reward와 nextstate에 대한 Q-value(discounted)의 합으로 표현
    target_Q_values = target_Q_values.reshape(-1,1) # 뒤의 Q_values와 차원 맞춰줌
    mask = tf.one_hot(actions, n_outputs) # 각 경험 별 action을 one-hot encoding 형태로 만들어줌 (각 행이 경험)
    with tf.GradientTape() as tape: # 자동 미분
        all_Q_values = model((input_load,input_pv,input_others)) 
        Q_values = tf.reduce_sum(all_Q_values * mask, # mask를 곱해서, 각 경험별로 그 state에서 취하지 않은 action에 대해서는 Q-value에 0이 곱해지도록 함
                                 axis=1, keepdims=True) # 각 행 별 합을 구함, 위의 masking 덕분에 그 state에서 취한 action에 대한 Q-value가 됨, keepdims를 True로 설정해 2차원 행렬 형태 유지
        loss = tf.reduce_mean(keras.losses.mean_squared_error(target_Q_values,Q_values)) # 평균제곱오차 계산
    grads = tape.gradient(loss, model.trainable_variables) # loss 함수를 model의 trainable variables 전체에 대해 미분
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) # Adam optimizer로 parameter update 수행, zip으로 gradient와 parameter pair를 맞춰줌


### 훈련 및 검증 수행

profits_test = []
elapsedtime_test = []
count_step_fortrain = 0
max_return_test = -np.inf

for epoch in range(100): # epoch 수 설정
    if epoch > 0: # 맨 첫 번째 epoch에서는 훈련을 시작하지 않고 buffer를 채움, 두 번째 epoch부터는 buffer에 sample들이 채워졌으므로 훈련함
        start_time = time.time()
    
    ### 훈련 (2년치)
    hour = 24 # 시작시점
    energy_batt = initialenergy_batt  # 배터리 내 에너지의 초기값
    
    for step in range(len(load_train)-25): # 데이터 내의 마지막 시점이 nextstate에만 포함될 때까지 (대신 termination state는 따로 없음)
        count_step_fortrain += 1
        epsilon = max(1 - epoch / 50, 0.1) # epsilon을 1에서 시작해 조금씩 선형적으로 줄임, 이 경우 초반에는 거의 다 exploration이므로 한 epoch가 매우 빨리 계산되나, 중반을 넘어가면 거의 exploitation이며 이 때 매 step마다 DNN을 call하므로 느려짐
        energy_batt, reward, action = play_one_step(load_train,PV_prod_train,hour,energy_batt,epsilon,training=True) # energy_batt가 반복 갱신됨
        hour += 1 # 다음 시간으로  

        if epoch > 0 and count_step_fortrain > period_step_fortrain: # 훈련 주기, 매 시간마다 train하면 epoch당 시간이 너무 길어지고 overfitting 우려도 있어, 매 24시간마다 훈련함, 첫 epoch에서는 replay buffer를 채우기만 하고, 나머지 99 epoch 동안 심층신경망 가중치를 업데이트함
                training_step(batch_size) # DNN 훈련을 위한 Gradient Descent 수행
                count_step_fortrain = 0


    ### 검증 (1년치)  
    if epoch > 0:
        testcase_actions = []
        testcase_battenergy = []
        epoch_return_test = 0 # return (각 시점별 수익의 총 합) 초기화
        
        hour = 24 # 시작시점   
        energy_batt = initialenergy_batt # 배터리 내 에너지의 초기값
        
        for step in range(len(load_test)-24): # 데이터 내의 마지막 시점이 nextstate에만 포함될 때까지 (대신 termination state는 따로 없음)
            energy_batt, reward, action = play_one_step(load_test,PV_prod_test,hour,energy_batt,0,training=False) # energy_batt가 반복 갱신됨
            testcase_actions.append(action)
            testcase_battenergy.append(energy_batt)
            epoch_return_test += reward # 누적보상 계산
            hour += 1 # 다음 시간으로    
        
        profits_test.append(epoch_return_test) # 각 epoch별 총 수익 로그 저장
        with open('trajectory_profit_test_dqn.txt', 'w') as f:
            for line in profits_test:
                f.write(f"{line}\n")
                
        if max_return_test < epoch_return_test: # Validation set에서의 총 수익의 최대값 갱신시마다 저장 (unlearn하게 되더라도 중간에 best performance였던 모델을 남김)
            max_return_test = epoch_return_test
            model.save_weights('trainedmodel_dqn.h5') # 모델 가중치 저장
                    
            with open('trajectory_actions_test_dqn.txt', 'w') as f: # Validation case에서의 시간별 action 로그 저장
                    for line in testcase_actions:
                        f.write(f"{line}\n")
            
            with open('trajectory_battenergy_test_dqn.txt', 'w') as f: # Validation case에서의 시간별 배터리 내 저장된 에너지 로그 저장
                    for line in testcase_battenergy:
                        f.write(f"{line}\n")
                        
        elapsed_time = time.time() - start_time   
        elapsedtime_test.append(elapsed_time)
        print("Validation: profit of epoch {} is {}, maximum profit is {}".format(epoch,round(epoch_return_test,2),round(max_return_test,2)))
        print('one epoch 수행에 {}초 걸렸습니다'.format(round(elapsed_time,2)))
        with open('trajectory_time_test_dqn.txt', 'w') as f: # 각 epoch별 소요시간 로그 저장
            for line in elapsedtime_test:
                f.write(f"{line}\n")