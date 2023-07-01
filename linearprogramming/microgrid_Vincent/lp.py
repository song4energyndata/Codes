from cvxopt import matrix, spmatrix, sparse, glpk
import numpy as np
import time

load_peak = 2
pv_peak = 12

eff_h2 = 0.65 # 수소-전기 변환효율

capa_batt = 15 # 배터리 용량 (겉보기용량이 아닌, SOC 상하한 고려한 실용량이라 가정)
eff_batt = 0.9 # 배터리 충방전 효율
initialenergy_batt = 0.0

price_h2 = 0.1 # 수소에너지의 가격 (Euro/kWh)
cost_loss = 2 # loss of load penalty (Euro/kWh)
maxrate_h2 = 1.1 # 계통으로부터의 송전/ 계통으로의 수전 의 상한 (kW)


data_load = np.loadtxt("load_test.txt")
data_pv = np.loadtxt("pv_test.txt")
p_load = data_load[24:] * load_peak
p_pv = data_pv[24:] * pv_peak
t=len(p_load)


def block_eye(size):
    return spmatrix(1,range(size),range(size)) 

def block_zeros(row,col):
    return sparse(matrix(np.zeros((row,col))))

def block_ones(row,col):
    return sparse(matrix(np.ones((row,col))))

def block_batt(size):
    return sparse([[block_zeros(size,1)],[block_eye(size)]]) - sparse([[block_eye(size)],[block_zeros(size,1)]])


Aeq_balance = sparse([[-block_eye(t)], [block_eye(t)], [-block_eye(t)], [block_eye(t)], [block_zeros(t,t+1)], [block_eye(t)], [-block_eye(t)]])
beq_balance = matrix(p_load-p_pv,tc='d')

Aeq_batterydynamic = sparse([[block_zeros(t,t)],[block_zeros(t,t)],[-eff_batt*block_eye(t)],[block_eye(t)/eff_batt],[block_batt(t)],[block_zeros(t,t)],[block_zeros(t,t)]])
beq_batterydynamic = block_zeros(t,1)

lowerbounds = matrix([[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[matrix([initialenergy_batt])],[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t)]]) 
upperbounds = matrix([[maxrate_h2*block_ones(1,t)],[maxrate_h2*block_ones(1,t)],[np.inf*block_ones(1,t)],[np.inf*block_ones(1,t)],[matrix([initialenergy_batt])],[capa_batt*block_ones(1,t)],[np.inf*block_ones(1,t)],[np.inf*block_ones(1,t)]]) 

A_lowerbound = -block_eye(7*t+1)
b_lowerbound = -lowerbounds.T

A_upperbound = block_eye(7*t+1)
b_upperbound = upperbounds.T

c = matrix([[price_h2*eff_h2*block_ones(1,t)],[-price_h2/eff_h2*block_ones(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t+1)],[-cost_loss*block_ones(1,t)],[block_zeros(1,t)]])

Aeq = sparse([Aeq_balance,Aeq_batterydynamic])
beq = matrix([beq_balance,beq_batterydynamic])

A = sparse([A_lowerbound, A_upperbound])
b = matrix([b_lowerbound, b_upperbound])


start_time = time.time()
x=glpk.lp(-c,A,b,Aeq,beq,options={ 'msg_lev': 'GLP_MSG_ON'})
elapsed_time = time.time() - start_time

profit = (c*x[1])[0]
print(profit)

p_send = np.array(x[1][0:t])
p_receive = np.array(x[1][(t):(2*t)]) # Python에서는 t+1:2*t+1 이 아님에 주의
p_ch = np.array(x[1][(2*t):(3*t)])
p_disch = np.array(x[1][(3*t):(4*t)])


import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
font_legend = font_manager.FontProperties(family='Cambria',
                                   style='normal', size=12)

starthour = 24*211+1
endhour = starthour + 72
axis_x = np.arange(starthour,endhour).reshape(-1,1)
axis_x_onedim = np.arange(starthour,endhour)


fig, ax = plt.subplots(figsize=(9,3))
#plt.plot(axis_x,p_pv[(starthour-1):(endhour-1)]*capa_pv, linewidth =0)
plt.plot(axis_x,p_load[(starthour-1):(endhour-1)],color='black', linewidth =2,label="Power load")
#plt.plot(axis_x,p_grid[(starthour-1):(endhour-1)]+p_pv[(starthour-1):(endhour-1)].reshape(-1,1)*capa_pv, linewidth =0)
ax.fill_between(axis_x_onedim, np.zeros((72,)), p_pv[(starthour-1):(endhour-1)], alpha=0.8, color='yellow',label="PV power",linewidth=0)
ax.fill_between(axis_x_onedim, p_pv[(starthour-1):(endhour-1)], p_receive[(starthour-1):(endhour-1)].reshape(-1)-p_send[(starthour-1):(endhour-1)].reshape(-1)+p_pv[(starthour-1):(endhour-1)], alpha=0.7, color='blue', label="Power from grid",linewidth=0)
ax.fill_between(axis_x_onedim, p_receive[(starthour-1):(endhour-1)].reshape(-1)-p_send[(starthour-1):(endhour-1)].reshape(-1)+p_pv[(starthour-1):(endhour-1)], p_receive[(starthour-1):(endhour-1)].reshape(-1)-p_send[(starthour-1):(endhour-1)].reshape(-1)+p_pv[(starthour-1):(endhour-1)]+p_disch[(starthour-1):(endhour-1)].reshape(-1), alpha=0.8, color='red', label="Battery discharge",linewidth=0)
