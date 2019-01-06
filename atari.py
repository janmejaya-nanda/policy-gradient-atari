import gym
import numpy as np

from agent import Agent
from constants import INPUT_SHAPE, TRAINING_FREQUENCY, REPLAY_START_LENGTH
from utils import get_next_state
from utils import preprocess_observation


class Atari(object):
    def __init__(self, environment='MsPacmanDeterministic-v4'):
        self.env = gym.make(environment)
        print('ACTIONS ', self.env.action_space.n)
        self.agent = Agent(
            n_actions=self.env.action_space.n,
            input_shape=INPUT_SHAPE
        )

    def run(self):
        episodes = 0

        while True:
            obs = preprocess_observation(self.env.reset())
            current_state = np.asarray([obs for _ in range(INPUT_SHAPE[2])])

            # Episodic loop
            done = False
            time_step = 0
            while done:
                action = self.agent.get_action(current_state)

                # observe reward and next state
                obs, reward, done, _ = self.env.step(action)
                obs = preprocess_observation(obs)
                next_state = get_next_state(current_state, obs)

                # Store transition in replay memory
                clipped_reward = np.clip(reward, -1, 1)  # Clip the reward
                self.agent.add_experience(
                    current_state=current_state,
                    next_state=next_state,
                    reward=clipped_reward,
                    action=action,
                    done=done
                )

                # Train the agent
                if time_step % TRAINING_FREQUENCY == 0 and len(self.agent.experiences) >= REPLAY_START_LENGTH:
                    self.agent.train()

                # Linear epsilon annealing
                if len(self.agent.experiences) >= REPLAY_START_LENGTH:
                    self.agent.update_epsilon()
                time_step += 1

            episodes += 1

if __name__ == '__main__':
    Atari().run()
