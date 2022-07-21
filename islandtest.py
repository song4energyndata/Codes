from cvxopt import matrix, spmatrix, sparse, glpk
import numpy as np
import time

penetration = 0.7
eff_batt = 0.94
cost_diesel = 328
eps = 0.0001

t = 8760

demand_island = np.loadtxt("demand.txt")  # 텍스트 파일을 배열로 불러옴 (큰따옴표 필요함에 주의)
re_output = np.loadtxt("reoutput.txt")
cost_cap = np.loadtxt("capcost.txt")
totaldemand = matrix(np.sum(demand_island))

pvoutput = matrix(re_output[0:t, :1])
wtoutput = matrix(re_output[0:t, 1:2])

block_eye = spmatrix(1, range(t), range(t))  # t x t 단위행렬
block_zeros = sparse(matrix(np.zeros((t, t))))  # t x t 영행렬. np.zeros 옆 괄호 2겹임에 주의
block_onevec = sparse(matrix(np.ones((t, 1))))  # t * 1 1벡터
block_zerovec = sparse(matrix(np.zeros((t, 1))))  # t* 1 0벡터
block_batt = sparse([[block_eye], [block_zerovec]]) - sparse(
    [[block_zerovec], [block_eye]]
)
# print(block_eye)
# print(block_zeros)
# print(block_batt)


Aeq_balance = sparse(
    [
        [block_eye],
        [block_eye],
        [-block_eye],
        [block_zeros],
        [block_zerovec],
        [-block_eye],
        [pvoutput],
        [wtoutput],
        [block_zerovec],
    ]
)
# print(Aeq_balance)
beq_balance = matrix(demand_island[0:t], tc="d")

Aeq_batterydynamic = sparse(
    [
        [block_zeros],
        [-1 / eff_batt * block_eye],
        [eff_batt * block_eye],
        [block_batt],
        [block_zeros],
        [block_zerovec],
        [block_zerovec],
        [block_zerovec],
    ]
)
# print(Aeq_batterydynamic)
beq_batterydynamic = block_zerovec

Aeq_batteryinitial = sparse(
    [
        [block_zerovec.T],
        [block_zerovec.T],
        [block_zerovec.T],
        [matrix([1])],
        [block_zerovec.T],
        [block_zerovec.T],
        [sparse([[0], [0], [0]])],
    ]
)
beq_batteryinitial = matrix([0])

A_maximumstored = sparse(
    [
        [block_zeros],
        [block_zeros],
        [block_zeros],
        [block_zerovec],
        [block_eye],
        [block_zeros],
        [block_zerovec],
        [block_zerovec],
        [-block_onevec],
    ]
)
b_maximumstored = block_zerovec
# print(A_maximumstored)

A_penetration = sparse(
    [
        [block_onevec.T],
        [block_zerovec.T],
        [block_zerovec.T],
        [block_zerovec.T],
        [block_zerovec.T],
        [sparse(matrix(np.zeros((1, 4))))],
    ]
)
b_penetration = matrix([(1 - penetration) * totaldemand])
# print(A_penetration)

A_lowerbound = spmatrix(-1, range(5 * t + 4), range(5 * t + 4))
b_lowerbound = matrix(np.zeros((5 * t + 4, 1)))
# print(A_lowerbound)

c = matrix(
    [
        [cost_diesel * block_onevec.T],
        [eps * block_onevec.T],
        [eps * block_onevec.T],
        [matrix([0])],
        [block_zerovec.T],
        [block_zerovec.T],
        [matrix([cost_cap[0]])],
        [matrix([cost_cap[1]])],
        [matrix([cost_cap[5]])],
    ]
)  # 이거 sparse로 하면 셧다운되는데 matrix로 하면 잘 됨...왜지..
# print(c)

Aeq = sparse([Aeq_balance, Aeq_batterydynamic, Aeq_batteryinitial])
# print(Aeq)
beq = matrix([beq_balance, beq_batterydynamic, beq_batteryinitial])

A = sparse([A_maximumstored, A_penetration, A_lowerbound])
# print(A)
b = matrix([b_maximumstored, b_penetration, b_lowerbound])

start_time = time.time()
(status, x) = glpk.ilp(c, A, b, Aeq, beq, I=set([43802]))
elapsed_time = time.time() - start_time

print(
    "태양광 {}kW, 풍력 {}기, 배터리 {}kWh입니다".format(
        round(x[43801]), round(x[43802]), round(x[43803])
    )
)
print("{}초 걸렸습니다".format(round(elapsed_time, 2)))
print((c * x)[0])
