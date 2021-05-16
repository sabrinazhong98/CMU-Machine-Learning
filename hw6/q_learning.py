import sys
from environment import MountainCar
import numpy as np

class LinearModel:
    def __init__(self, lr, env, mode):
      
        self.lr = lr
        self.w = np.zeros([env.action_space, env.state_space])
        self.dw = np.zeros([env.action_space, env.state_space])
        self.bias = np.array(0, dtype = 'float64')
        self.mode = mode
    
    
    def cal_qs(self, state,current = True):
        
        if current == True:
            
            self.qvals = np.array([np.dot(state.T, self.w[action])+ self.bias for action in range(3)], dtype = 'float64')

        else:
            self.qvals_next = np.array([np.dot(state.T, self.w[action])+ self.bias for action in range(3)], dtype = 'float64')
        
    def update(self, state_current,state_next, action, r, gamma):
    
        self.qcur = np.dot(state_current, self.w[action]) + self.bias
        self.dw[action] = state_current
        
        
        self.w -= self.lr * (self.qcur - (r + gamma * np.max(self.qvals_next))) * self.dw
        self.bias -= self.lr * (self.qcur - (r+ gamma*np.max(self.qvals_next)))
 
    def zero_grad(self):
        self.dw = np.zeros([env.action_space, env.state_space])



def get_action(optimal, epsilon ):
        p = np.random.random()
        
        if p <= epsilon:
            
            return np.random.choice(3)
        else:
            
            return optimal

def changestate(state, mode, state_space):
    if mode == 'tile':
        keys = state.keys()
        state = np.zeros(state_space)
        for k in keys:
            state[k] = 1
        
    elif mode == 'raw':
        state = np.array(list(state.values()))
    return state

class QLearningAgent:
    def __init__(self, env, mode, gamma, lr, epsilon):
        self.mode = mode
        self.gamma= gamma
        self.lr = lr
        self.epsilon = epsilon
        self.env = env

    def training(self,episodes, max_iterations):
          
        all_rewards = [0]
        linear = LinearModel(self.lr, self.env, self.mode)
        
        for e in range(episodes):
            state = self.env.reset()
            state = changestate(state, self.mode,self.env.state_space)
            
            done = False
            rewards = 0
            
            for s in range(max_iterations):
                linear.cal_qs(state)

                optimal = np.argmax(linear.qvals)
                action = get_action(optimal, self.epsilon)
          
                
                nextstate, reward, done = self.env.step(action)
                
                nextstate = changestate(nextstate, self.mode,self.env.state_space)
 
                rewards += reward
                
                #gradient
                linear.cal_qs(nextstate, current = False)
                
                linear.update(state, nextstate, action, reward, self.gamma)
                linear.zero_grad()
                                
                state = nextstate
                
                
                if done == True:
                    break
                   
            all_rewards.append(rewards)
        return linear.w, linear.bias, all_rewards
    
        


def weights_to_list(weights, bias):
    wgtlist = [float(bias)]
    h = weights.shape[1]
    
    for x in range(h):
        temp_list = [w for w in weights[:,x]]
        wgtlist.extend(temp_list)
    return wgtlist   

if __name__ == "__main__":
    """
    mode = "tile"
    weight_out = 'weightout'
    returns_out = 'returns.out'
    episodes = 20
    max_iterations = 20  #maximum length of an episode
    epsilon = 0
    gamma = 0.99 # discount factor
    lr = 0.00005 # learning rate
    """
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5]) #maximum length of an episode
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7]) # discount factor
    lr = float(sys.argv[8]) # learning rate
    
    env = MountainCar(mode = mode)
   # linear = LinearModel(lr, env, mode)
    agent = QLearningAgent(env, mode, gamma, lr, epsilon)
    weights, bias, rewards = agent.training(episodes, max_iterations)
    weights_list = weights_to_list(weights, bias)
 
    
    with open(weight_out, 'w') as f:
        f.writelines("%s\n" % line for line in weights_list)
        
    with open(returns_out, 'w') as f:
        f.writelines("%s\n" % line for line in rewards[1:])
   
    
