import numpy as np


class CarpoolEnv:

    def __init__(self, distances, start=0, goal=0, capacity=4, delay=0.0):
        """
        Args:
            distances (ndarray): the distance matrix
            start (int): driver's initial location index in the distance matrix
            goal (int): goal's index in the distance matrix
            capacity (int): how many passengers the car fits
            delay (float): maximum delay (in seconds or meters based on distance matrix units)
            reward_mult (float): factor that is multiplied with reward to give it greater significance
        """

        self.dist = distances
        self.n_states = distances.shape[0]
        self.n_actions = distances.shape[1]

        self.start = start
        self.goal = goal
        self.d_thr = self.dist[self.start][self.goal] + delay  # set distance threshold
        self.capacity = capacity

        self.state = self.start
        self.action = self.start
        self.done = False
        self.passengers = 0
        self.dist_covered = 0.0
        self.cumulated_reward = 0.0
        self.valid_actions = [a for a in range(self.n_actions)
                              if a != self.start and a != self.goal]
        self.route = [self.start]

    def reset(self):
        self.state = self.start
        self.action = self.start
        self.done = False
        self.passengers = 0
        self.dist_covered = 0.0
        self.cumulated_reward = 0.0
        self.valid_actions = [a for a in range(self.n_actions)
                              if a != self.start and a != self.goal]
        self.route = [self.start]

        return self.state, self.valid_actions

    def sample(self):
        action = np.random.choice(self.valid_actions)
        return action

    def step(self, action):
        if self.done:
            return None, 0.0, True, None

        self.valid_actions.remove(action)
        self.action = action

        reward = 0.0

        # does selecting the point lead to surpassing the distance threshold
        # or not?
        if self.dist_covered + self.dist[self.state][self.action] + \
                self.dist[self.action][self.goal] <= self.d_thr:

            reward += 1.0 - self.dist[self.state][self.action] / self.d_thr
            self.passengers += 1
            self.dist_covered += self.dist[self.state][self.action]
            self.route.append(action)
            self.state = self.action
        else:
            self.done = True

        # check if car is full or there are no more actions to take
        if self.passengers >= self.capacity or not self.valid_actions:
            self.done = True

        if self.done:
            reward += 1.0 - self.dist[self.state][self.goal] / self.d_thr
            self.dist_covered += self.dist[self.state][self.goal]
            self.route.append(self.goal)
            self.state = None

        self.cumulated_reward += reward

        return self.state, reward, self.done, self.valid_actions
