# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.layers import Input, Dense, Conv1D, concatenate, Flatten
import time

### hyperparameters

alpha_lr = 0.00005 # temperature parameter에 대한 learning rate
actor_lr = 0.0005
critic_lr = 0.001
tau = 0.05
alpha_initial = 0.1 # temperature parameter의 초기값
target_entropy = -tf.constant(1, dtype=tf.float32) # 1은 DoF of action
rewardscalefactor = 1 # SAC가 reward scale에 민감하므로 잘 선택해야 함: reward가 너무 낮으면 너무 exploration만 해버리고, reward가 너무 높으면 처음엔 학습이 빨리 되는 것 같아 보이지만 poor local minima로 수렴함이 알려짐
buffer_size = 20000
batch_size = 32
discount_factor = 0.98
period_step_fortrain = 24



### microgrid system data

PV_prod_train = np.load('BelgiumPV_prod_train.npy')
PV_prod_test  = np.load('BelgiumPV_prod_test.npy')  

load_train = np.load('example_nondeterminist_cons_train.npy')
load_test  = np.load('example_nondeterminist_cons_test.npy')

prate_h2 = 1.1
eff_h2 = 0.65

capa_batt = 15
eff_batt = 0.9
initialenergy_batt = 0.0

price_h2 = 0.1
cost_loss = 2

load_peak = 2
pv_peak = 12

inputlen_load = 24
inputlen_pv = 24



### Neural net configuration

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()         
        self.input_load = Input(shape=(inputlen_load,1))
        self.input_pv = Input(shape=(inputlen_pv,1))
        self.hidden_conv_load = Conv1D(16, 2, activation='relu')(self.input_load)
        self.hidden_conv_pv = Conv1D(16, 2, activation='relu')(self.input_pv)
        self.concat_conv = concatenate([self.hidden_conv_load,self.hidden_conv_pv],axis=2)
        self.hidden_conv_concat = Conv1D(16,2,activation='relu')(self.concat_conv)
        self.flatten_conv = Flatten()(self.hidden_conv_concat)
        self.input_others = Input(shape=(1))
        self.input_action = Input(shape=(1))
        self.concat_all = concatenate([self.flatten_conv,self.input_others,self.input_action]) 
        self.hidden_dense_1 = Dense(50,activation='relu')(self.concat_all)
        self.hidden_dense_2 = Dense(20,activation='relu')(self.hidden_dense_1)
        self.output_qval = Dense(1)(self.hidden_dense_2)
        self.model = keras.Model(inputs=[self.input_load,self.input_pv,self.input_others,self.input_action], outputs=[self.output_qval])     
    def call(self, input_load,input_pv,input_others,input_action):
        return self.model((input_load,input_pv,input_others,input_action))

class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()       
        self.input_load = Input(shape=(inputlen_load,1))
        self.input_pv = Input(shape=(inputlen_pv,1))
        self.hidden_conv_load = Conv1D(16, 2, activation='relu')(self.input_load)
        self.hidden_conv_pv = Conv1D(16, 2, activation='relu')(self.input_pv)
        self.concat_conv = concatenate([self.hidden_conv_load,self.hidden_conv_pv],axis=2)
        self.hidden_conv_concat = Conv1D(16,2,activation='relu')(self.concat_conv)
        self.flatten_conv = Flatten()(self.hidden_conv_concat) 
        self.input_others = Input(shape=(1))
        self.concat_all = concatenate([self.flatten_conv,self.input_others]) 
        self.hidden_dense_1 = Dense(50,activation='relu')(self.concat_all)
        self.hidden_dense_2 = Dense(20,activation='relu')(self.hidden_dense_1)
        self.output_action_mean = Dense(1,activation='linear')(self.hidden_dense_2) # 정규분포의 평균
        self.output_action_stddev = Dense(1,activation='linear')(self.hidden_dense_2) # 정규분포의 표준편차
        self.model = keras.Model(inputs=[self.input_load,self.input_pv,self.input_others], outputs=[self.output_action_mean,self.output_action_stddev])     
    def call(self, input_load,input_pv,input_others,training=True):        
        mu, logsigma = self.model((input_load,input_pv,input_others)) # 정규분포의 평균, 로그표준편차
        sigma = tf.exp(logsigma) # 로그표준편차를 표준편차로 변환
        dist = tfp.distributions.Normal(mu, sigma) # 정규분포 정의
        if training == True: # Training 시
            action_temp = dist.sample() # 정규분포로부터 임의로 샘플 뽑아서 action 결정 
        else: # Validation 시
            action_temp = mu # Validation 시에는 mean action으로 (deterministic action)
        action = tf.tanh(action_temp) # tanh activation function을 이용한 bounded action 구현
        logprob_temp = dist.log_prob(action_temp)
        logprob = logprob_temp - tf.reduce_sum(tf.math.log(1 - action**2 + 1e-16), axis=1,
                                         keepdims=True) # log-probability 반환
        return action, logprob
        

critic_one_learning = Critic()
critic_one_target = Critic()
critic_one_learning.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))
critic_one_target.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))
critic_one_target.set_weights(critic_one_learning.get_weights())

critic_two_learning = Critic()
critic_two_target = Critic()
critic_two_learning.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))
critic_two_target.compile(optimizer=keras.optimizers.Adam(learning_rate=critic_lr))
critic_two_target.set_weights(critic_two_learning.get_weights()) 

actor_learning = Actor()
actor_target = Actor()
actor_learning.compile(optimizer=keras.optimizers.Adam(learning_rate=actor_lr))
actor_target.compile(optimizer=keras.optimizers.Adam(learning_rate=actor_lr))
actor_target.set_weights(actor_learning.get_weights())

alpha = tf.Variable(alpha_initial, dtype=tf.float32) # temperature parameter 정의
alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

actor_weight_temp = actor_learning.get_weights()
critic_one_weight_temp = critic_one_learning.get_weights()
critic_two_weight_temp = critic_two_learning.get_weights()


from collections import deque
replay_buffer = deque(maxlen=buffer_size)

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(4)]
    return states, actions, rewards, next_states



def play_one_step(profile_load,profile_pv,hour,energy_batt,training=False):
        
    state = np.concatenate( (profile_load[hour-inputlen_load:hour], profile_pv[hour-inputlen_pv:hour], np.array([energy_batt])) )
    
    action, _ = actor_learning(profile_load[hour-inputlen_load:hour].reshape(-1,inputlen_load,1), profile_pv[hour-inputlen_pv:hour].reshape(-1,inputlen_pv,1), np.array([energy_batt]).reshape(-1,1),training=training)
    action = action.numpy()[0][0] 
        
    p_h2_send = 0
    p_h2_receive = 0
    if action > 0:
        p_h2_receive = prate_h2*action
    else:
        p_h2_send = -prate_h2*action
    p_load = profile_load[hour]*load_peak
    p_pv = profile_pv[hour]*pv_peak
    energy_batt = energy_batt*capa_batt
        
    # p_curtail = 0
    p_loss = 0
    
    p_net_beforebatt = p_pv - p_load + p_h2_receive - p_h2_send
    
    if p_net_beforebatt >= 0:
        if capa_batt >= energy_batt + p_net_beforebatt*eff_batt:
            energy_batt_after = energy_batt + p_net_beforebatt*eff_batt
        else:
            energy_batt_after = capa_batt
            # p_curtail = (energy_batt + p_net_beforebatt*eff_batt - capa_batt)/eff_batt 
    else: 
        if energy_batt*eff_batt >= -p_net_beforebatt:
            energy_batt_after = energy_batt + p_net_beforebatt/eff_batt
        else:
            energy_batt_after = 0
            p_loss = -p_net_beforebatt - energy_batt*eff_batt           
    
    reward = price_h2*p_h2_send*eff_h2 - price_h2*p_h2_receive/eff_h2 - cost_loss*p_loss
    energy_batt_after = energy_batt_after/capa_batt
    
    if training == True:
        next_state = np.concatenate( (profile_load[hour-(inputlen_load-1):hour+1], profile_pv[hour-(inputlen_pv-1):hour+1], np.array([energy_batt_after])) )         
        replay_buffer.append((state, action, reward*rewardscalefactor, next_state))
    return energy_batt_after, reward, action



def training_step(batch_size,actorupdate=True):
    
    states, actions, rewards, next_states = sample_experiences(batch_size)
        
    input_load = states[:,0:inputlen_load].reshape(-1,inputlen_load,1) 
    input_pv = states[:,inputlen_load:(inputlen_load+inputlen_pv)].reshape(-1,inputlen_pv,1)
    input_others = states[:,(inputlen_load+inputlen_pv)].reshape(-1,1)
    
    input_load_next = next_states[:,0:inputlen_load].reshape(-1,inputlen_load,1)
    input_pv_next = next_states[:,inputlen_load:(inputlen_load+inputlen_pv)].reshape(-1,inputlen_pv,1)
    input_others_next = next_states[:,(inputlen_load+inputlen_pv)].reshape(-1,1)
        
    actions_by_target, logprobs_actions_target = actor_target(input_load_next,input_pv_next,input_others_next)# action 뿐 아니라 로그확률도 불러옴, 로그확률은 entropy term에 사용됨
    
    Q_values_one_by_target = critic_one_target(input_load_next,input_pv_next,input_others_next,actions_by_target)
    Q_values_two_by_target = critic_two_target(input_load_next,input_pv_next,input_others_next,actions_by_target)
    Q_values_min_by_target = tf.math.minimum(Q_values_one_by_target, Q_values_two_by_target)
    y = tf.stop_gradient(rewards + discount_factor * (Q_values_min_by_target - alpha*logprobs_actions_target)) # entropy term이 추가됨됨
    
    with tf.GradientTape() as tape:
        Q_values_one = critic_one_learning(input_load,input_pv,input_others,actions.reshape(-1,1)) #
        loss_critic_one = tf.reduce_mean(keras.losses.mean_squared_error(y,Q_values_one)) 
    grads = tape.gradient(loss_critic_one, critic_one_learning.trainable_variables)
    critic_one_learning.optimizer.apply_gradients(zip(grads, critic_one_learning.trainable_variables))
    
    with tf.GradientTape() as tape:
        Q_values_two = critic_two_learning(input_load,input_pv,input_others,actions.reshape(-1,1))
        loss_critic_two = tf.reduce_mean(keras.losses.mean_squared_error(y,Q_values_two))
    grads = tape.gradient(loss_critic_two, critic_two_learning.trainable_variables)
    critic_two_learning.optimizer.apply_gradients(zip(grads, critic_two_learning.trainable_variables))
    
    if actorupdate == True:    
        with tf.GradientTape() as tape:
            actions_by_learner, logprobs_actions = actor_learning(input_load,input_pv,input_others)
            Q_values_one = critic_one_learning(input_load,input_pv,input_others,actions_by_learner)
            Q_values_two = critic_two_learning(input_load,input_pv,input_others,actions_by_learner)
            Q_values_min = tf.math.minimum(Q_values_one, Q_values_two)
            loss_actor = tf.reduce_mean(alpha*logprobs_actions - Q_values_min)
        grads = tape.gradient(loss_actor, actor_learning.trainable_variables)
        actor_learning.optimizer.apply_gradients(zip(grads, actor_learning.trainable_variables))
        
        with tf.GradientTape() as tape: # temperature parameter의 업데이트
            actions_by_learner, logprobs_actions = actor_learning(input_load,input_pv,input_others)
            loss_alpha = tf.reduce_mean(-alpha*(logprobs_actions+target_entropy))
        grads = tape.gradient(loss_alpha, [alpha])
        alpha_optimizer.apply_gradients(zip(grads, [alpha]))
        
        actor_weights = actor_learning.weights
        target_actor_weights = actor_target.weights
        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
        actor_target.set_weights(target_actor_weights)
        
        critic_one_weights = critic_one_learning.weights
        critic_two_weights = critic_two_learning.weights
        target_critic_one_weights = critic_one_target.weights
        target_critic_two_weights = critic_two_target.weights
        for i in range(len(critic_one_weights)):
            target_critic_one_weights[i] = tau * critic_one_weights[i] + (1 - tau) * target_critic_one_weights[i]
            target_critic_two_weights[i] = tau * critic_two_weights[i] + (1 - tau) * target_critic_two_weights[i]
        critic_one_target.set_weights(target_critic_one_weights)    
        critic_two_target.set_weights(target_critic_two_weights)    



profits_test = []
elapsedtime_test = []
max_return_test = -np.inf
count_step_fortrain = 0
bool_fortrain = False


for epoch in range(100):
    if epoch > 0:
        start_time = time.time()
    
    ### Train for each epoch
    
    hour = 24
    energy_batt = initialenergy_batt  
    
    for step in range(len(load_train)-25):
        count_step_fortrain += 1      
        energy_batt, reward, action = play_one_step(load_train,PV_prod_train,hour,energy_batt,training=True)
        hour += 1     

        if epoch > 0 and count_step_fortrain > period_step_fortrain: 
                training_step(batch_size,actorupdate=bool_fortrain)
                count_step_fortrain = 0
                bool_fortrain = not bool_fortrain                                   
                actor_weight_temp = actor_learning.get_weights()
                critic_one_weight_temp = critic_one_learning.get_weights()
                critic_two_temp = critic_two_learning.get_weights()
                    
    ### Validation for each epoch  
    if epoch > 0:
        testcase_actions = []
        testcase_battenergy = []
        epoch_return_test = 0
        
        hour = 24
        energy_batt = initialenergy_batt
        
        for step in range(len(load_test)-24):
            energy_batt, reward, action = play_one_step(load_test,PV_prod_test,hour,energy_batt,training=False)
            testcase_actions.append(action)
            testcase_battenergy.append(energy_batt)
            epoch_return_test += reward
            hour += 1
        
        profits_test.append(epoch_return_test)   
        with open('trajectory_profit_test_sac.txt', 'w') as f:
            for line in profits_test:
                f.write(f"{line}\n")
    
        if max_return_test < epoch_return_test:
            max_return_test = epoch_return_test
            actor_learning.save_weights('actor_trainedmodel_sac.h5')
            critic_one_learning.save_weights('critic_one_trainedmodel_sac.h5')
            critic_two_learning.save_weights('critic_two_trainedmodel_sac.h5')
                    
            with open('trajectory_actions_test_sac.txt', 'w') as f:
                    for line in testcase_actions:
                        f.write(f"{line}\n")
            
            with open('trajectory_battenergy_test_sac.txt', 'w') as f:
                    for line in testcase_battenergy:
                        f.write(f"{line}\n")
                        
        elapsed_time = time.time() - start_time   
        elapsedtime_test.append(elapsed_time)
        print("Validation: profit of epoch {} is {}, maximum profit is {}".format(epoch,round(epoch_return_test,2),round(max_return_test,2)))
        print('one epoch 수행에 {}초 걸렸습니다'.format(round(elapsed_time,2)))
        print('Temperature parameter 값은 {}입니다.'.format(alpha.numpy()))
        with open('trajectory_time_test_sac.txt', 'w') as f:
            for line in elapsedtime_test:
                f.write(f"{line}\n")