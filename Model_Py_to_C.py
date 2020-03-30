'''
Convert a pytorch model trained in Python to be compatible with loading in C++

A PyTorch model’s journey from Python to C++ is enabled by Torch Script, a representation of a PyTorch model that can be understood, compiled and serialized by the Torch Script compiler
'''
import torch
import torchvision
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

# An instance of your model.
DQN = QNetwork()
DQN.load_state_dict(torch.load('checkpointPID.pth'))

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 4)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(DQN, example)

# To serialize the model, so that there is no longer any dependency on Python
traced_script_module.save("traced_DQN_modelPID.pth")

