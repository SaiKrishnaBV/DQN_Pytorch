'''
To train a neural network to fit the data obtained from the experience of the agent with an environment
'''
import random
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LR = 0.001
EPOCHS = 20001
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
		#dim - is the dimension along which the softmax will be computed
		# -1 -> Allows the function to infer the right dimension


DQN = QNetwork()
optimizer = optim.Adam(DQN.parameters(), lr = 0.001)
loss_fn = nn.NLLLoss()

train_data = np.load('PIDdata.npy',allow_pickle = True)
x_data = []
y_data = []
for i in train_data:
	x_data.append(i[0])
	if(i[1] == [0,1]):
		y_data.append(1)
	else:
		y_data.append(0)
Xdata = torch.from_numpy(np.asarray(x_data)).float()

#Ylabel needs to match the required format for the defined loss function
Ylabel = torch.from_numpy(np.asarray(y_data)).view(1,-1)[0]
#Ylabel is reshaped as a 1D tensor - one row containing all labels

for epoch in range(1,EPOCHS):
	optimizer.zero_grad()
	Ypred = DQN(Xdata)

	loss = loss_fn(Ypred, Ylabel)
	loss.backward()

	optimizer.step()

	'''
	output of the neural network is probability, to choose the actual action
	_,pred = DQN(X).data.max(1)
	'''

	if(epoch % 100 == 0):
		print(str(epoch) + " : " + str(loss.data.item()))
		#savename = 'checkpoint' + str(epoch) + '.pth'
		

'''
checkpoint = {'model': QNetwork(),
              'state_dict': DQN.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
'''
torch.save(DQN.state_dict(),'checkpointPID.pth')
