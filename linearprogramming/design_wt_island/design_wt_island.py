from cvxopt import matrix, spmatrix, sparse, glpk
import numpy as np
import time

penetration = 0.6
eff_batt = 0.94
cost_diesel= 300 # 디젤발전기로 발전된 전기 1kWh당 투입연료비
c_rate_batt = 0.5
initialenergy_batt = 0
penaltycoeff = 0.001 # 동시 충/방전 오류를 막기 위한 penalty term의 계수 (너무 낮으면 오류 생김)

r = 0.05 # 화폐가치 하락에 대한 할인율

initialcost_pv = 3000000 # 태양광 1kW당 초기투자비
initialcost_wt = 80000000 # 풍력 1대당 초기투자비
initialcost_batt = 900000 # 배터리 1kWh당 초기투자비 (SOC 고려한 실용량 가정)
mtncost_pv = initialcost_pv * 0.02 # 태양광 1kW당 유지보수비 (매년 발생) 
mtncost_wt = initialcost_wt * 0.02 # 풍력 1대당 유지보수비 (매년 발생)
mtncost_batt = initialcost_batt * 0.01 # 배터리 1kWh당 유지보수비 (매년 발생)


t=8760

demand_island = np.loadtxt("load_hourly_island.txt") 
renewable_output = np.loadtxt("renewables_hourly_island.txt")  
totaldemand = matrix(np.sum(demand_island))

pvoutput = matrix(renewable_output[0:t,0]) # 1kW 태양광 패널의 시간별 전기 출력
wtoutput = matrix(renewable_output[0:t,2]) # 대상 규격 풍력터빈 1대의 시간별 전기 출력

def block_eye(size):
    return spmatrix(1,range(size),range(size)) 

def block_zeros(row,col):
    return sparse(matrix(np.zeros((row,col))))

def block_ones(row,col):
    return sparse(matrix(np.ones((row,col))))

def block_batt(size):
    return sparse([[block_zeros(size,1)],[block_eye(size)]]) - sparse([[block_eye(size)],[block_zeros(size,1)]])


Aeq_balance = sparse([[block_eye(t)], [-block_eye(t)], [block_eye(t)], [block_zeros(t,t+1)], [-block_eye(t)], [pvoutput],[wtoutput],[block_zeros(t,1)]])
beq_balance = matrix(demand_island[0:t],tc='d')

Aeq_batterydynamic = sparse([[block_zeros(t,t)],[-eff_batt*block_eye(t)],[block_eye(t)/eff_batt],[block_batt(t)],[block_zeros(t,t)],[block_zeros(t,1)],[block_zeros(t,1)],[block_zeros(t,1)]])
beq_batterydynamic = block_zeros(t,1)

A_maximumstored = sparse([[block_zeros(t,t)],[block_zeros(t,t)],[block_zeros(t,t)],[block_zeros(t,1)],[block_eye(t)],[block_zeros(t,t)],[block_zeros(t,1)],[block_zeros(t,1)],[-block_ones(t,1)]])
b_maximumstored = block_zeros(t,1)

A_maximumcharge = sparse([[block_zeros(t,t)],[block_eye(t)],[block_zeros(t,t)],[block_zeros(t,t+1)],[block_zeros(t,t)],[block_zeros(t,1)],[block_zeros(t,1)],[-c_rate_batt*block_ones(t,1)]])
b_maximumcharge = block_zeros(t,1)

A_maximumdischarge = sparse([[block_zeros(t,t)],[block_zeros(t,t)],[block_eye(t)],[block_zeros(t,t+1)],[block_zeros(t,t)],[block_zeros(t,1)],[block_zeros(t,1)],[-c_rate_batt*block_ones(t,1)]])
b_maximumdischarge = block_zeros(t,1)

A_penetration = sparse([[block_ones(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t+1)],[block_zeros(1,t)],[block_zeros(1,3)]])
b_penetration = matrix([(1-penetration)*totaldemand])

lowerbounds = matrix([[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,t)],[matrix([initialenergy_batt])],[block_zeros(1,t)],[block_zeros(1,t)],[block_zeros(1,3)]]) 
upperbounds = matrix([[np.inf*block_ones(1,t)],[np.inf*block_ones(1,t)],[np.inf*block_ones(1,t)],[matrix([initialenergy_batt])],[np.inf*block_ones(1,t)],[np.inf*block_ones(1,t)],[np.inf*block_ones(1,3)]]) 

A_lowerbound = -block_eye(5*t+4)
b_lowerbound = -lowerbounds.T

A_upperbound = block_eye(5*t+4)
b_upperbound = upperbounds.T

Aeq = sparse([Aeq_balance,Aeq_batterydynamic])
beq = matrix([beq_balance,beq_batterydynamic])

A = sparse([A_maximumstored, A_maximumcharge, A_maximumdischarge, A_penetration, A_lowerbound, A_upperbound])
b = matrix([b_maximumstored, b_maximumcharge, b_maximumdischarge, b_penetration, b_lowerbound, b_upperbound])

npvfactor = (1/(1+r))*(1-(1/(1+r))**20)/(1-(1/(1+r))) # 초항이 (1+r)^{-1}, 마지막항이 (1+r)^{20}인 등비수열의 합
costfun_pv = initialcost_pv + mtncost_pv * npvfactor # 초기투자비 + 매년 유지보수비(npvfactor를 곱해 현가화)
costfun_wt = initialcost_wt + mtncost_wt * npvfactor # 초기투자비 + 매년 유지보수비(npvfactor를 곱해 현가화)
costfun_batt = initialcost_batt * (1 + (1/(1+r))**10) + mtncost_batt * npvfactor # 초기투자비 + 10년 후 재투자비(현가화) + 매년 유지보수비 (현가화)
costfun_diesel = cost_diesel * block_ones(1,t) * npvfactor  # 전기 사용량 요금 (현가화)


c = matrix([[costfun_diesel],[penaltycoeff*block_ones(1,t)],[penaltycoeff*block_ones(1,t)],[block_zeros(1,t+1)],[block_zeros(1,t)],[matrix([costfun_pv])],[matrix([costfun_wt])],[matrix([costfun_batt])]]) # 이거 sparse로 하면 셧다운되는데 matrix로 하면 잘 됨...왜지..

idx_pv = (t+t+t+(t+1)+t+1)-1
idx_wt = (t+t+t+(t+1)+t+2)-1 # 변수 벡터 x에서 풍력터빈 대수 변수의 인덱스
idx_batt = (t+t+t+(t+1)+t+3)-1

start_time = time.time()
(status,x)=glpk.ilp(c,A,b,Aeq,beq,I=set([idx_wt])) # 풍력터빈 대수 변수를 정수로 제한
elapsed_time = time.time() - start_time

p_ch = np.array(x[(t):(2*t)])
p_disch = np.array(x[(2*t):(3*t)])
indicator_error = (p_ch.reshape(1,-1) @ p_disch.reshape(-1,1))[0][0]

if indicator_error > 1:
    print('경고: 같은 시간에 충전과 방전이 동시에 양수가 되는 오류가 발생했습니다.')
print('태양광 {}kW, 풍력 {}기, 배터리 {}kWh입니다'.format(x[idx_pv],x[idx_wt],x[idx_batt]))
print('{}초 걸렸습니다'.format(round(elapsed_time,2)))
print((c*x)[0] - penaltycoeff*np.sum(p_ch+p_disch)) # 총 비용 (penalty term의 값 제함)
