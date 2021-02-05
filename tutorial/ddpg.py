"""
This tutorial uses a simple implementation of Deep Deterministic Policy Gradient (DDPG), 
derived from keras examples, refactored to work with this tutorial.

This implementation uses tensorflow, and keras wrapers for the neural networks to do 
the heavy lifting of the implementation for us. 
For more examles visit: https://keras.io/examples/rl/ddpg_pendulum/
"""

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from tutorial.exploration import OUActionNoise
from tutorial.rpm import Buffer

if __name__ == '__main__':
    problem = "Pendulum-v0"
    env = gym.make(problem)

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
