import numpy as np

# p.183~184 example
def comparison_normal_smapling():
    x = np.array([1, 2, 3]) # 확률 변수
    pi = np.array([0.1, 0.1, 0.8]) # 확률 분포

    # 기댓값의 참값 계산
    e = np.sum(x * pi)
    print("true value (E_pi[x]): ", e)

    # 몬테 카를로법 계산
    n = 100 # 샘플 수
    smaples = []
    for _ in range(n):
        s = np.random.choice(x, p=pi)
        smaples.append(s)
    mean = np.mean(smaples)
    var = np.var(smaples)
    print(f"Monte Calro (E_pi[x]): {mean}, Var[x]: {var}")


# p.185 example
def importance_sampling():
    x = np.array([1, 2, 3]) # 확률 변수
    pi = np.array([0.1, 0.1, 0.8]) # 확률 분포
    b = np.array([1/3, 1/3, 1/3]) # 확률 분포
    n = 100 # 샘플 개수
    samples = []
    
    for _ in range(n):
        idx = np.arange(len(b)) # b의 인덱스([0, 1, 2])
        i = np.random.choice(idx, p=b) # b를 이용한 샘플링
        s = x[i]
        rho = pi[i] / b[i] # 가중치
        samples.append(s * rho) # 샘플 데이터에 가중치를 곱해 저장
    mean = np.mean(samples)
    var = np.var(samples)
    print(f"Importance Sampling (E_pi[x]): {mean}, Var[x]: {var}")

# p.187 example
def revised_importance_sampling():
    x = np.array([1, 2, 3]) # 확률 변수
    pi = np.array([0.1, 0.1, 0.8]) # 확률 분포
    #b = np.array([1/3, 1/3, 1/3]) # 확률 분포
    b = np.array([0.2, 0.2, 0.6]) # 분포 변경
    n = 100 # 샘플 개수
    samples = []
    
    for _ in range(n):
        idx = np.arange(len(b)) # b의 인덱스([0, 1, 2])
        i = np.random.choice(idx, p=b) # b를 이용한 샘플링
        s = x[i]
        rho = pi[i] / b[i] # 가중치
        samples.append(s * rho) # 샘플 데이터에 가중치를 곱해 저장
    mean = np.mean(samples)
    var = np.var(samples)
    print(f"Importance Sampling (E_pi[x]): {mean}, Var[x]: {var}")

    
    

        
        