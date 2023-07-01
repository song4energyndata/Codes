# -*- coding: utf-8 -*-

from cvxopt import matrix, spmatrix, sparse, glpk
import numpy as np
import matplotlib.pyplot as plt

def block_eye(size):
    return spmatrix(1,range(size),range(size)) 

def block_zeros(row,col):
    return sparse(matrix(np.zeros((row,col))))

def block_batt(size):
    return sparse([[block_zeros(size,1)],[block_eye(size)]]) - sparse([[block_eye(size)],[block_zeros(size,1)]])

def block_ones(row,col):
    return sparse(matrix(np.ones((row,col))))

p_load = 300*np.ones((24,1))

t = 24

block_zeros_storage = sparse(matrix(np.zeros((t,t+1)))) # t x (t+1) 영행렬
Aeq_balance = sparse([ [block_eye(t)], [-block_eye(t)], [block_eye(t)], [block_zeros(t,t+1)] ]) # 블록들은 왼쪽에서부터 수전, 충전, 방전, 배터리내에너지
beq_balance = matrix(p_load,tc='d') # tc='d'는 double형(실수) 의미

eff_batt = 0.9 # 충전/방전 효율은 90%로 가정

Aeq_batteryenergy = sparse([[block_zeros(t,t)],[-eff_batt*block_eye(t)],[1/eff_batt*block_eye(t)],[block_batt(t)]])
beq_batteryenergy = block_zeros(t,1)

Aeq = sparse([Aeq_balance,Aeq_batteryenergy]) # ([]) 안의 요소들에 []를 안 붙이면 vstack, []를 붙이면 hstack으로 생각
beq = matrix([beq_balance,beq_batteryenergy])

initialenergy_batt = 0
lowerbounds = matrix([[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[matrix([initialenergy_batt])],[block_zeros(1,t)]]) 

capacity_batt = 100
c_rate_batt = 0.5
block_onevec = sparse(matrix(np.ones((t,1)))) # t x 1 일벡터 (모든 원소가 1)
upperbounds = matrix([[np.inf*block_ones(1,t)],[c_rate_batt*capacity_batt*block_ones(1,t)],[c_rate_batt*capacity_batt*block_ones(1,t)],[matrix([initialenergy_batt])],[capacity_batt*block_ones(1,t)]]) 

A_lowerbound = -block_eye(4*t+1)
b_lowerbound = -lowerbounds.T # lowerbounds는 행벡터였음, 이를 열벡터로 바꿈
A_upperbound = block_eye(4*t+1)
b_upperbound = upperbounds.T # upperbounds는 행벡터였음, 이를 열벡터로 바꿈

A = sparse([A_lowerbound, A_upperbound]) # lower bound는 부등호 방향 때문에 마이너스
b = matrix([b_lowerbound, b_upperbound]) # lower bound는 부등호 방향 때문에 마이너스, b들은 전치하였음에 주의

eleccost_offpeak = (87.3+9+5)*1.137 # 경부하 시간의 전력요금 (9와 5는 각각 기후환경요금단가와 연료비조정단가, 1.137은 부가가치세 및 전력산업기반기금 비율)
eleccost_mid = (140.2+9+5)*1.137 # 중간부하 시간의 전력요금
eleccost_peak = (222.3+9+5)*1.137 # 최대부하 시간의 전력요금
eleccost = np.hstack([eleccost_offpeak*np.ones((1,8)),
                      eleccost_mid*np.ones((1,3)),
                      eleccost_peak*np.ones((1,1)),
                      eleccost_mid*np.ones((1,1)),
                      eleccost_peak*np.ones((1,5)),
                      eleccost_mid*np.ones((1,4)),
                      eleccost_offpeak*np.ones((1,2))]) # 24차원 벡터, 각 원소는 해당 시간의 전기요금
c = matrix([[matrix(eleccost)],[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t+1)]]) # 목적함수에서 변수들의 계수들

x = glpk.lp(c,A,b,Aeq,beq) # 선형계획법 문제를 품
totalcost = (c*x[1])[0] # 24시간 동안의 총 전기요금
p_grid = np.array(x[1][0:24]) # 시간별 수전
p_charge = np.array(x[1][24:48]) # 시간별 충전
p_discharge = np.array(x[1][48:72]) # 시간별 방전
e_batt = np.array(x[1][72:]) # 시간별 배터리 내 에너지 (0번째 시간 포함)

import matplotlib.font_manager as font_manager
font_legend = font_manager.FontProperties(family='Cambria',
                                   style='normal', size=12)

plt.figure(figsize=(6,3))
plt.plot(np.arange(1,25).reshape(-1,1),p_load,label="Power load")
plt.plot(np.arange(1,25).reshape(-1,1),p_grid,label="Power from grid")
plt.plot(np.arange(0,25).reshape(-1,1),e_batt,label="Energy in battery")
plt.xlabel("Time [hour]",fontsize=12,font="Cambria")
plt.ylabel("Electricity [kWh]",fontsize=14,font="Cambria")
plt.xlim((0,25))
plt.ylim((0,400))
plt.xticks(np.arange(1,25))
plt.grid(True)
plt.legend(prop=font_legend,loc="lower right",ncol=1)
plt.savefig('result_24hour.png',dpi=200,bbox_inches="tight")


