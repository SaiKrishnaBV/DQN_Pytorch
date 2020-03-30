'''
To get data from the agent's experience with the environment using a random policy
'''
import random
import numpy as np
import gym

REQ_EPISODE_COUNT = 100
REQ_SCORE = 80
training_data = []
EPI_COUNT = 0
done = False
score = 0

env = gym.make('CartPole-v0').env
env.reset()
print("Initializing the procedure")
while(EPI_COUNT <= REQ_EPISODE_COUNT):
	episode_memory = []
	obs = env.reset()
	while(not done):
		action = random.randrange(0,2)
		observation, reward, done, info = env.step(action)

		episode_memory.append([obs,action])
		obs = observation
		score += reward
	done = False
	#print(score)
	if(score >= REQ_SCORE):
		del(episode_memory[-10:])
		EPI_COUNT += 1
		for data in episode_memory:
			if(data[1] == 1):
				output = [0,1]
			else:
				output = [1,0]
			training_data.append([data[0],output])
		print(EPI_COUNT)
	score = 0
DATA = np.asarray(training_data)
np.save('Qdata.npy',DATA)
