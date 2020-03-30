'''
To evaluate a neural network, on the environment
'''
import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HIDDEN_LAYER = 4
class QNetwork(nn.Module):
	def __init__(self):
		'''
		super(QNetwork, self).__init__()
		in python 2
		'''
		super().__init__()
		self.l1 = nn.Linear(4, HIDDEN_LAYER)
		self.l2 = nn.Linear(HIDDEN_LAYER, 2)
		
	def forward(self, X):
		X = F.sigmoid(self.l1(X))
		X = self.l2(X)
		return F.log_softmax(X, dim=-1) 

DQN = QNetwork()
DQN.load_state_dict(torch.load('checkpointPID_smallNetwork.pth'))
env = gym.make('CartPole-v0').env
env.reset()
TR = []
for i in range(100):
	obs = env.reset()
	obs = torch.from_numpy(obs).float()
	done = False
	Reward = 0
	while(not(done)):
		action = DQN(obs)
		tst, action = torch.max(action, 0)
		#env.render()
		obs, reward, done, info = env.step(action.item())
		obs = torch.from_numpy(obs)
		obs = obs.float()
		Reward += reward
		if(Reward > 1000000):
			done = True
	TR.append(Reward)
	print(Reward)
print(sum(TR)/len(TR))
