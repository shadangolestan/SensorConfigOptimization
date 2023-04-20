import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class DQN:
    def __init__(self, env) -> None:
        self.env = env
        self.states = self.env.observation_space.shape
        self.actions = self.env.action_space.n

        # del self.model
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape = self.states))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.actions, activation='linear'))
        return model

    def model_summary(self):
        print(self.model.summary())

    def build_agent(self):
        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit = 50000, window_length = 1)
        self.dqn = DQNAgent(model= self.model, 
                    memory= memory, 
                    policy= policy,
                    nb_actions = self.actions, 
                    nb_steps_warmup = 10, 
                    target_model_update = 1e-2)

        self.dqn.compile(Adam(lr = 1e-3), metrics=['mae'])
        
        
    
    
    # self.dqn.fit(self.env, nb_steps=1000, visualize=False, verbose = 1)
        

    def test(self):
        return self.dqn.test(self.env, nb_episodes=100, visualize=False)




