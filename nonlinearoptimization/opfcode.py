# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize


v_bus = np.arange(0,5) # 인덱스 0~4는 각 bus의 전압크기
del_bus = np.arange(5,10,1) # 인덱스 5~9는 각 bus의 전압위상각
p_bus = np.arange(10,15,1) # 인덱스 10~14는 각 bus 내 발전기의 유효전력 (발전기가 없으면 0)
q_bus = np.arange(15,20,1) # 인덱스 15~19는 각 bus 내 발전기의 무효전력 (발전기가 없으면 0)
phi_transf = 20 # 인덱스 20은 bus 3과 4 간 변압기의 위상차
mag_transf = 21 # 인덱스 21은 bus 3과 5 간 변압기의 전압비

n_bus = 5

y_series = np.zeros((n_bus,n_bus),dtype=complex) # 각 전선의 series admittance
y_sh_branch = np.zeros((n_bus,n_bus),dtype=complex) # 각 전선의 shunt admittance
y_sh_bus = np.zeros((n_bus),dtype=complex) # 각 bus의 shunt admittance

y_series[0,1]= 1/(0+0.3j) # bus 1과 2를 잇는 전선의 series admittance
y_series[0,2] = 1/(0.023+0.145j)
y_series[1,3] = 1/(0.006+0.032j)
y_series[2,3] = 1/(0.02+0.26j)
y_series[2,4] = 1/(0+0.32j)
y_series[3,4] = 1/(0+0.5j)
y_sh_branch[0,2] = (0+0.04j) # bus 1과 3을 잇는 전선의 shunt admittance
y_sh_branch[1,3] = (0+0.01j)
y_sh_bus[1] = 0+0.3j # bus 2의 shunt admittance
y_sh_bus[2] = 0.05+0j
    
load_bus = np.array([0+0j, 0+0j, 0+0j, 0.9+0.4j, 0.239+0.129j]) # bus 4와 5의 전력 부하

def nonlconfun(x): # 비선형 제약조건의 좌변을 변수들의 함수로 계산

    a_tx = np.ones((n_bus,n_bus),dtype=complex) # 변압비와 위상차 phasor        
    a_tx[2,3] = 1*np.exp(1j*x[phi_transf]) # bus 3과 4 간 변압기의 위상차는 최적화로 도출 필요
    a_tx[2,4] = x[mag_transf]*np.exp(1j*0) # bus 3과 5 간 변압기의 전압비는 최적화로 도출 필요
    
    y_full = np.zeros((n_bus,n_bus),dtype=complex) # admittance matrix
    
    for fr in range(n_bus):
        y_full[fr,fr] = y_sh_bus[fr] # Y_ii
        for to in range(n_bus):
            y_full[fr,fr] = y_full[fr,fr] + (y_series[fr,to] + 0.5*y_sh_branch[fr,to])/(np.abs(a_tx[fr,to])**2) + (y_series[to,fr] + 0.5*y_sh_branch[to,fr])
            if to != fr: # Y_ik
                y_full[fr,to] = y_full[fr,to] - y_series[fr,to]/np.conjugate(a_tx[fr,to]) - y_series[to,fr]/a_tx[to,fr] 
    
    ceq = np.zeros((n_bus*2)) # 제약조건의 좌변 값을 담을 벡터
    
    for m in range(n_bus): # Power flow equation
        ceq[m] = np.real(load_bus[m]) - x[p_bus[m]] # 인덱스 0, 1, ..., m-1은 유효전력
        ceq[n_bus+m] = np.imag(load_bus[m]) - x[q_bus[m]] # 인덱스 m, m+1, ..., 2m-1은 무효전력
        for k in range(n_bus):
            ceq[m] = ceq[m] + x[v_bus[m]]*x[v_bus[k]]*(np.real(y_full[m,k])*np.cos(x[del_bus[m]]-x[del_bus[k]]) + np.imag(y_full[m,k])*np.sin(x[del_bus[m]]-x[del_bus[k]])) # real power at bus m
            ceq[n_bus+m] = ceq[n_bus+m] + x[v_bus[m]]*x[v_bus[k]]*(np.real(y_full[m,k])*np.sin(x[del_bus[m]]-x[del_bus[k]]) - np.imag(y_full[m,k])*np.cos(x[del_bus[m]]-x[del_bus[k]])) # reactive power at bus m
    
    return ceq # 제약조건의 좌변 값을 반환


def objfun(x):
    cost = 0.35*x[p_bus[0]] + 0.2*x[p_bus[2]] + 0.4*x[p_bus[2]]**2 + 0.3*x[p_bus[3]] + 0.5*x[p_bus[3]]**2
    return cost

bounds = [(1,1),
        (0.95,1.05),
        (0.95,1.05),
        (0.95,1.05),
        (0.95,1.05),
        (0,0),
        (-np.pi,np.pi), # 각도의 단위가 radian임에 주의
        (-np.pi,np.pi),
        (-np.pi,np.pi),
        (-np.pi,np.pi),
        (-10,10),
        (0,0),
        (0.1,0.4),
        (0.05,0.4),
        (0,0),
        (-10,10),
        (0,0),
        (-0.2,0.2),
        (-0.2,0.2),
        (0,0),
        (-np.pi/6,np.pi/6),
        (0.95,1.05)]

initialpoint = (np.array(bounds)[:,0]+np.array(bounds)[:,1])/2

nonlcon = NonlinearConstraint(nonlconfun, np.zeros((n_bus*2,)), np.zeros((n_bus*2,))) # 모든 변수들에 대해 우변의 하한과 상한이 모두 0, 즉 우변이 0인 등호조건

sol = minimize(objfun,initialpoint,bounds=bounds,constraints=nonlcon) # solver로 OPF 문제 풀기

solution = np.round(np.multiply(sol.x, # 전압 위상각의 단위를 radian에서 degree로 변환
                       np.array([1,1,1,1,1,180/np.pi,180/np.pi,180/np.pi,180/np.pi,180/np.pi,1,1,1,1,1,1,1,1,1,1,180/np.pi,1])),3)
print(solution[v_bus[1]],solution[v_bus[2]],solution[v_bus[3]],solution[v_bus[4]],"\n",
      solution[del_bus[1]],solution[del_bus[2]],solution[del_bus[3]],solution[del_bus[4]],"\n",
      solution[p_bus[0]],solution[p_bus[2]],solution[p_bus[3]],"\n",
      solution[q_bus[0]],solution[q_bus[2]],solution[q_bus[3]],"\n",
      solution[phi_transf],solution[mag_transf])   
