import numpy as np
from collections import defaultdict

from .common import GridWorld

def greedy_probs(Q, state, action_size=4):
    qs = [Q[(state,action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    
    action_probs = {action: 0 for action in range(action_size)}
    # 이 시점에서 action_probs는 {0: 0, 1: 0, 2: 0, 3: 0} 형태
    action_probs[max_action] = 1
    return action_probs # 탐욕 행동을 취하는 확률 분포 반환

def eps_greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)
    
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    # 이 시점에서 action_probs는 {0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4} 형태
    action_probs[max_action] += (1 - epsilon)
    return action_probs # ε-탐욕 행동을 취하는 확률 분포 반환

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []
        
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()
        
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.cnts[key] += 1
            
            # 개선 전
            self.Q[key] += (G - self.Q[key]) / self.cnts[key] # self.Q 갱신
            
            # 개선 후
            alpha = 0.1
            self.Q[key] += (G - self.Q[key]) * alpha # self.Q 갱신
            
            # state 정책 탐욕화
            self.pi[state] = greedy_probs(self.Q, state, self.action_size)

            
class McAgentBetter:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1 # e
        self.alpha = 0.1 # a
        self.action_size = 4
        
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        #self.cnts = defaultdict(lambda: 0)
        self.memory = []
        
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()
        
    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.Q[key] += (G - self.Q[key]) * self.alpha # self.Q 갱신
            
            # state 정책 탐욕화
            self.pi[state] = eps_greedy_probs(self.Q, state, self.epsilon)
            
def mcagent_train():
    env = GridWorld()
    agent = McAgentBetter()
    episodes = 10000
    
    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add(state, action, reward)
            
            if done:
                agent.update()
                break
            state = next_state
            
    env.render_q(agent.Q)
    
    