import numpy as np

def dice():
    ps = {2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36, 7: 
          6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36}
    
    V = 0
    for x, p in ps.items():
        V += x * p
    print(V)
    

def sample_dice(dices=2):
    x = 0
    for _ in range(dices):
        # 주사위 수만큼 반복하여, 각 주사위 눈을 합하는 연산
        x += np.random.randint(1, 7) # 1 ~ 6의 난수 생성
    return x

def ev_sample_dice(dices=2, trial=1000):
    # trial: 샘플링 횟수
    
    samples = []
    for _ in range(trial):
        s = sample_dice(dices)
        samples.append(s)
    #V = sum(samples) / len(samples) 
    #V = np.mean(samples)
    V = sum(samples) / trial
    return V

def increment_ev_sample_dice(dices=2, trial=1000):
    V, n = 0, 0
    
    for idx in range(trial):
        s = sample_dice(dices)
        n += 1
        V += (s - V) / n # 또는 V = V + (s - V) / n
        if idx % 100 == 0:
            print(f"step: {idx}, V: {V}")
    return V
