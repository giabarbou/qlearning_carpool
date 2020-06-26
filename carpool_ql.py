import numpy as np
import matplotlib.pyplot as plt
from itertools import count


class QLearning:
    def __init__(self, env, lr=0.01, eps=0.1, gamma=1.0, eps_min=None,
                 eps_decay=0.0):

        self.env = env
        self.lr = lr
        self.eps = eps
        self.gamma = gamma

        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.num_episodes = 0
        self.Q = np.zeros([env.n_states, env.n_actions])
        self.rewards_all_episodes = []

    def run(self, num_episodes, log_step=0, seed=None):
        if num_episodes <= 0:
            return

        if seed is not None:
            np.random.seed(seed)

        if self.eps_min is None or self.eps_min > self.eps or self.eps_decay == 0:
            self.eps_min = self.eps
        eps_max = self.eps

        self.num_episodes = num_episodes
        self.rewards_all_episodes = []

        for i_ep in range(self.num_episodes):

            state, valid_actions = self.env.reset()
            rewards_current_episode = 0.0

            for t in count():
                valid_actions = np.asarray(valid_actions)

                # explore or exploit
                if np.random.uniform(0, 1) < self.eps:
                    action = self.env.sample()
                else:
                    idx = np.argmax(self.Q[state, valid_actions])
                    action = valid_actions[idx]
                
                # take action and observe reward and next state
                new_state, reward, done, valid_actions = self.env.step(action)

                # update Q values according to sampled reward
                self.Q[state, action] = self.Q[state, action] + \
                                        self.lr * (reward + self.gamma * np.max(self.Q[new_state, valid_actions]) -
                                            self.Q[state, action])
                
                state = new_state
                rewards_current_episode += reward

                if done:
                    break

            # decrease epsilon according to decay rate
            self.eps = self.eps_min + (eps_max - self.eps_min) * \
                       np.exp(-self.eps_decay * i_ep)
            self.rewards_all_episodes.append(rewards_current_episode)

            if log_step and i_ep % log_step == 0:
                print("Episode: ", i_ep)
                print("Route: ", self.env.route)
                print("Reward: ", '%.2f' % rewards_current_episode)
                print("Distance: ", self.env.dist_covered)
                print("Epsilon: ", '%.2f' % self.eps)
                print('')

        return self.Q

    def select_route(self):
        # select optimal route according to learnt Q values
        
        state, valid_actions = self.env.reset()
        rewards = 0.0
        for t in count():
            valid_actions = np.asarray(valid_actions)

            idx = np.argmax(self.Q[state, valid_actions])
            action = valid_actions[idx]

            new_state, reward, done, valid_actions = self.env.step(action)

            state = new_state
            rewards += reward

            if done:
                break

        return self.env.route, self.env.dist_covered, rewards

    def plot(self, smoothness=100):
        if not self.rewards_all_episodes:
            return

        data = self.rewards_all_episodes
        n = int(smoothness)
        if self.num_episodes >= n > 1:
            data = np.convolve(self.rewards_all_episodes, np.ones(n) / n, mode='valid')

        plt.clf()
        plt.plot(data)
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.title('Rewards per episode (smoothed)')
        plt.show()
