# -*- coding: utf-8 -*-

from cvxopt import matrix, spmatrix, sparse, glpk
import numpy as np
import matplotlib.pyplot as plt

p_load = np.loadtxt("load_hourly_bldg.txt")
p_pv = np.loadtxt("pv_hourly_bldg.txt") # 1kW 패널의 시간별 발전량
eleccost = np.loadtxt("eleccost_hourly_bldg.txt")
eleccost += (9+5) # 기후환경요금 9원/kWh, 연료비조정요금 5원/kWh 반영
eleccost_demandcharge = 8320 # kW당 기본요금
taxrate = 1.137

eff_batt = 0.94
c_rate_batt = 0.5
initialenergy_batt = 0

r = 0.05 # 화폐가치 하락에 대한 할인율

initialcost_pv = 1600000 # 태양광 1kW당 초기투자비
initialcost_batt = 800000 # 배터리 1kWh당 초기투자비 (SOC 고려한 실용량 가정)
mtncost_pv = initialcost_pv * 0.02 # 태양광 1kW당 유지보수비 (매년 발생) 
mtncost_batt = initialcost_batt * 0.01 # 배터리 1kWh당 유지보수비 (매년 발생)

t = 8760

def block_eye(size):
    return spmatrix(1,range(size),range(size)) 

def block_zeros(row,col):
    return sparse(matrix(np.zeros((row,col))))

def block_ones(row,col):
    return sparse(matrix(np.ones((row,col))))

def block_batt(size):
    return sparse([[block_zeros(size,1)],[block_eye(size)]]) - sparse([[block_eye(size)],[block_zeros(size,1)]])


Aeq_balance = sparse([ [block_eye(t)], [-block_eye(t)], [block_eye(t)], [block_zeros(t,t+1)], [block_zeros(t,1)], [matrix(p_pv)], [block_zeros(t,1)]  ])
beq_balance = matrix(p_load,tc='d')

Aeq_batteryenergy = sparse([[block_zeros(t,t)],[-eff_batt*block_eye(t)],[1/eff_batt*block_eye(t)],[block_batt(t)], [block_zeros(t,3)]])
beq_batteryenergy = block_zeros(t,1)

Aeq_batteryenergy_lasthour = matrix([[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[matrix([1])],[block_zeros(1,t-1)],[matrix([-1])],[block_zeros(1,3)]])
beq_batteryenergy_lasthour = matrix([0])

Aeq = sparse([Aeq_balance, Aeq_batteryenergy, Aeq_batteryenergy_lasthour])
beq = matrix([beq_balance, beq_batteryenergy, beq_batteryenergy_lasthour])


A_maximumstored = sparse([[block_zeros(t,t)],[block_zeros(t,t)],[block_zeros(t,t)],[block_zeros(t,1)],[block_eye(t)],[block_zeros(t,2)],[-block_ones(t,1)]])
b_maximumstored = block_zeros(t,1)

A_maximumcharge = sparse([[block_zeros(t,t)],[block_eye(t)],[block_zeros(t,t)],[block_zeros(t,t+1)],[block_zeros(t,2)],[-c_rate_batt*block_ones(t,1)]])
b_maximumcharge = block_zeros(t,1)

A_maximumdischarge = sparse([[block_zeros(t,t)],[block_zeros(t,t)],[block_eye(t)],[block_zeros(t,t+1)],[block_zeros(t,2)],[-c_rate_batt*block_ones(t,1)]])
b_maximumdischarge = block_zeros(t,1)

A_maximumgrid = sparse([[block_eye(t)],[block_zeros(t,t)],[block_zeros(t,t)],[block_zeros(t,t+1)],[-block_ones(t,1)],[block_zeros(t,2)]])
b_maximumgrid = block_zeros(t,1)

lowerbounds = matrix([[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[matrix([initialenergy_batt])],[block_zeros(1,t)],[block_zeros(1,3)]])
A_lowerbound = -block_eye(4*t+4)
b_lowerbound = -lowerbounds.T

upperbounds = matrix([[np.inf*block_ones(1,t)],[np.inf*block_ones(1,t)],[np.inf*block_ones(1,t)],[matrix([initialenergy_batt])],[np.inf*block_ones(1,t)],[np.inf*block_ones(1,3)]]) 
A_upperbound = block_eye(4*t+4)
b_upperbound = upperbounds.T

A = sparse([A_maximumstored, A_maximumcharge, A_maximumdischarge, A_maximumgrid, A_lowerbound, A_upperbound])
b = matrix([b_maximumstored, b_maximumcharge, b_maximumdischarge, b_maximumgrid, b_lowerbound, b_upperbound])

npvfactor = (1/(1+r))*(1-(1/(1+r))**20)/(1-(1/(1+r))) # 초항이 (1+r)^{-1}, 마지막항이 (1+r)^{20}인 등비수열의 합
costfun_pv = initialcost_pv + mtncost_pv * npvfactor # 초기투자비 + 매년 유지보수비(npvfactor를 곱해 현가화)
costfun_batt = initialcost_batt * (1 + (1/(1+r))**10) + mtncost_batt * npvfactor # 초기투자비 + 10년 후 재투자비(현가화) + 매년 유지보수비 (현가화)
costfun_elec = taxrate * eleccost.reshape(1,-1) * npvfactor # 전기 사용량 요금 (현가화)
costfun_elec_demand = taxrate * eleccost_demandcharge * 12 * npvfactor # 전기 기본요금 (현가화), 매 월별이므로 12 곱함

c = matrix([[matrix(costfun_elec)],[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t+1)],[matrix([costfun_elec_demand])],[matrix([costfun_pv])],[matrix([costfun_batt])]]) 

x = glpk.lp(c,A,b,Aeq,beq)
totalcost = (c*x[1])[0]
capa_pv = np.array(x[1][-2]) # 태양광 용량
capa_batt = np.array(x[1][-1]) # 배터리 용량
p_grid = np.array(x[1][0:t]) # 시간별 수전
p_ch = np.array(x[1][(t):(2*t)]) # 시간별 충전, Python에서는 t+1:2*t+1 이 아님에 주의
p_disch = np.array(x[1][(2*t):(3*t)]) # 시간별 방전
print(capa_pv)
print(capa_batt)


import matplotlib.font_manager as font_manager
font_legend = font_manager.FontProperties(family='Cambria',
                                   style='normal', size=12)

starthour = 24*211+1
endhour = starthour + 72
axis_x = np.arange(starthour,endhour).reshape(-1,1)
axis_x_onedim = np.arange(starthour,endhour)

fig, ax = plt.subplots(figsize=(9,3))

plt.plot(axis_x,p_load[(starthour-1):(endhour-1)],color='black', linewidth =2,label="Power load")
ax.fill_between(axis_x_onedim, np.zeros((72,)), p_pv[(starthour-1):(endhour-1)]*capa_pv, alpha=0.8, color='yellow',label="PV power",linewidth=0)
ax.fill_between(axis_x_onedim, p_pv[(starthour-1):(endhour-1)]*capa_pv, p_grid[(starthour-1):(endhour-1)].reshape(-1)+p_pv[(starthour-1):(endhour-1)]*capa_pv, alpha=0.7, color='blue', label="Power from grid",linewidth=0)
ax.fill_between(axis_x_onedim, p_grid[(starthour-1):(endhour-1)].reshape(-1)+p_pv[(starthour-1):(endhour-1)]*capa_pv, p_grid[(starthour-1):(endhour-1)].reshape(-1)+p_pv[(starthour-1):(endhour-1)]*capa_pv+p_disch[(starthour-1):(endhour-1)].reshape(-1), alpha=0.8, color='red', label="Battery discharge",linewidth=0)

plt.xlim((starthour-1,endhour))
plt.ylim((0,2500))
plt.ylabel("Electricity use [kWh]",fontsize=14,font="Cambria")
plt.grid(True)
plt.legend(loc='lower center', bbox_to_anchor=(0.5,-0.3),ncol=4)
plt.savefig('result_year.png',dpi=200,bbox_inches="tight")