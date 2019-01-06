from random import random
from random import randint

import numpy as np

from constants import BATCH_SIZE, EPSILON_CHANGE_RATE, MINIMUM_EPSILON
from constants import EPSILON_START
from constants import GAMMA
from constants import REPLAY_MEMORY_LENGTH
from policy_network import PolicyNetwork


class Agent(object):
    def __init__(self, n_actions, input_shape):
        self.n_actions = n_actions
        self.epsilon = EPSILON_START
        self.experiences = []
        self.reward = 0
        self.policy_network = PolicyNetwork(
            n_actions=n_actions,
            input_shape=input_shape
        )

    def train(self):
        batch = self._get_sample_batch()
        self.policy_network.train(batch=batch)

    def get_action(self, state):
        if random() < self.epsilon:
            # explore
            return randint(0, self.n_actions - 1)
        else:
            # exploit best action
            actions = self.policy_network.predict(state=state)
            return np.argmax(actions)

    def add_experience(self, current_state, next_state, reward, action, done):
        if len(self.experiences) > REPLAY_MEMORY_LENGTH:
            self.experiences.pop(0)

        discounted_reward = reward + GAMMA * self.reward
        self.experiences.append({
            'current_state': current_state,
            'next_state': next_state,
            'action': action,
            'done': done,
            'reward': discounted_reward
        })
        self.reward = 0 if done else discounted_reward

    def update_epsilon(self):
        if self.epsilon - EPSILON_CHANGE_RATE > MINIMUM_EPSILON:
            self.epsilon -= EPSILON_CHANGE_RATE
        else:
            self.epsilon = MINIMUM_EPSILON

    def _get_sample_batch(self):
        return [self.experiences[i] for i in random.sample(range(0, REPLAY_MEMORY_LENGTH), BATCH_SIZE)]
