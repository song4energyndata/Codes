from cvxopt import matrix, glpk
import time


c = matrix(
    [0, -1], tc="d"
)  # -1 since we're maximising the 2nd variable, tc는 type code로 i면 정수, d면 실수(근데 그게 의미가 있나?)
G = matrix([[-1, 1], [3, 2], [2, 3], [-1, 0], [0, -1]], tc="d")
# G=sparse(matrix(G_np),tc='d') # numpy array를 행렬로 바꿀 수 있음, 단 tc='d'는 필수입력인듯?
# G=sparse([[-1,1],[3,2],[2,3],[-1,0],[0,-1]],tc='d') # sparse의 경우 tc는 i가 될 수 없음
h = matrix([1, 12, 12, 0, 0], tc="d")
# A=matrix([1,-1.1],(1,2),tc='d') # 괄호는 1행 2열짜리 행렬임을 의미(이거 안쓰면 열이 1개인 벡터가 되어버림)
# b=matrix([0],tc='d')

start_time = time.time()
(status, x) = glpk.ilp(c, G.T, h, I=set([0, 1]))  # I=set([0,1])은 1번째 & 2번째 변수가 정수라는 뜻일
# x=glpk.lp(c,G.T,h) # G와 h는 ineq (<=), A와 b는 eq에 대응, 단순 lb와 ub는 ineq에 포함시켜야 함
elapesed_time = time.time() - start_time

print(x)  # x벡터 산출 (x가 tuple이기 때문. x[0]은 optimal인데 의미는 모르겠음)
print(-c.T * x)  # 목적함수 값 산출 (여기서는 최대화였으므로 - 붙인거). numpy array를 바꾼다면 전치 없애야 함
