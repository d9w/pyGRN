from datetime import datetime
import numpy as np
import gym
import os
from PIL import Image

from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from .base import Problem
from pygrn.grns import DiffGRN
from pygrn.layer import GRNLayer, FixedGRNLayer


ATARI_INPUT_SHAPE = (84, 84)


class DQNProblem(Problem):

    def __init__(self, log_file, seed=0, learn=True, env_name='CartPole-v0',
                 nsteps=10000, warmup=10):
        self.log_file = log_file
        self.env = gym.make(env_name)
        self.nb_actions = self.env.action_space.n
        self.seed = seed
        self.learn = learn
        self.nsteps = nsteps
        self.warmup = warmup
        self.eval_count = 0
        self.generation = 0
        self.nin = 20
        self.nout = 20
        self.cacheable = False

    def generation_function(self, grneat, generation):
        self.generation = generation

    def get_model(self, grn):
        return Sequential()

    def eval(self, grn):
        self.eval_count += 1
        self.env.seed(self.seed+123)
        np.random.seed(self.seed+123)
        model = self.get_model(grn)

        memory = SequentialMemory(limit=self.nsteps,
                                  window_length=self.window_length)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=model, nb_actions=self.nb_actions,
                       memory=memory, nb_steps_warmup=self.warmup,
                       processor=self.processor, gamma=0.99,
                       train_interval=self.window_length, delta_clip=1.0,
                       target_model_update=1e-2, policy=policy,
                       custom_model_objects={'GRNLayer': GRNLayer,
                                             'FixedGRNLayer': FixedGRNLayer})
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        history = dqn.fit(self.env, nb_steps=self.nsteps, visualize=False,
                          verbose=0)
        final = dqn.test(self.env, nb_episodes=3, visualize=False)
        fit = 0.0
        for i in range(len(final.history['episode_reward'])):
            fit += final.history['episode_reward'][i]
        fit /= len(final.history['episode_reward'])
        if self.env == "Acrobot-v1":
            fit += 500
        elif self.env == "MountainCar-v0":
            fit += 200

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
        np.random.seed(self.seed+self.eval_count)
        return fit


class SLGym(DQNProblem):
    processor = None
    window_length = 1

    def __init__(self, *args, **kwargs):
        super(SLGym, self).__init__(*args, **kwargs)
        self.nin = np.prod(self.env.observation_space.shape)
        self.nout = self.nb_actions

    def get_model(self, grn):
        grn_str = str(grn)
        # model start
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        # model GRN layer
        layer = GRNLayer(grn_str, warmup_count=0, pmin=-10.0, pmax=10.0)
        if not self.learn:
            layer = FixedGRNLayer(grn_str, warmup_count=0, pmin=-10.0, pmax=10.0)
        model.add(layer)

        return model


class Gym(DQNProblem):
    processor = None
    window_length = 1

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


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(ATARI_INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == ATARI_INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


class Atari(DQNProblem):
    input_shape = ATARI_INPUT_SHAPE
    window_length = 4
    processor = AtariProcessor()

    def get_model(self, grn):
        grn_str = str(grn)
        input_shape = (self.window_length,) + self.input_shape
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=input_shape))
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.nin))
        model.add(Activation('relu'))
        layer = GRNLayer(grn_str, warmup_count=0, pmin=-10.0, pmax=10.0)
        if not self.learn:
            layer = FixedGRNLayer(grn_str, warmup_count=0, pmin=-10.0, pmax=10.0)
        model.add(layer)
        model.add(Dense(self.nout))
        model.add(Activation('relu'))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))
        return model
