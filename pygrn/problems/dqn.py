from datetime import datetime
import numpy as np
import gym
import os

from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from .base import Problem
from pygrn.grns import DiffGRN
from pygrn.layer import GRNLayer, FixedGRNLayer

class DQNProblem(Problem):

    def __init__(self, log_file, learn=True, env_name='CartPole-v0',
                 nsteps=10000, warmup=10):
        self.log_file = log_file
        self.env = gym.make(env_name)
        self.nb_actions = self.env.action_space.n
        self.learn = learn
        self.nsteps = nsteps
        self.warmup = warmup
        self.eval_count = 0
        self.generation = 0
        self.nin = 20
        self.nout = 20

    def generation_function(self, grneat, generation):
        self.generation = generation

    def get_model(self, grn):
        return Sequential()

    def eval(self, grn):
        self.eval_count += 1
        self.env.seed(123)
        model = self.get_model(grn)

        memory = SequentialMemory(limit=self.nsteps, window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=model, nb_actions=self.nb_actions,
                       memory=memory, nb_steps_warmup=self.warmup,
                       target_model_update=1e-2, policy=policy,
                       custom_model_objects={'GRNLayer': GRNLayer,
                                             'FixedGRNLayer': FixedGRNLayer})
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        history = dqn.fit(self.env, nb_steps=self.nsteps, visualize=False,
                          verbose=0)
        final = dqn.test(self.env, nb_episodes=1, visualize=False)
        fit = final.history['episode_reward'][0]

        with open(self.log_file, 'a') as f:
            for i in range(len(history.history['episode_reward'])):
                f.write('L,%s,%d,%d,%d,%f\n' % (
                    datetime.now().isoformat(),
                    self.generation, self.eval_count, i,
                    history.history['episode_reward'][i]))
            f.write('M,%s,%d,%d,%f\n' % (
                datetime.now().isoformat(),
                self.generation, self.eval_count, fit))

        del model
        K.clear_session()
        return fit

class Gym(DQNProblem):

    def get_model(self, grn):
        grn_str = str(grn)
        # model start
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(Dense(self.nin))
        model.add(Activation('relu'))

        # model GRN layer
        layer = GRNLayer(grn_str, warmup_count=0, pmin=-10.0, pmax=10.0)
        if not self.learn:
            layer = FixedGRNLayer(grn_str, warmup_count=0, pmin=-10.0, pmax=10.0)
        model.add(layer)

        # model LSTM layer
        # model.add(Reshape((1,self.nin)))
        # model.add(LSTM(self.nin))

        # model end part
        model.add(Dense(self.nout))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        return model
