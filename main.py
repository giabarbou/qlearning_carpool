import numpy as np
from carpool_env import CarpoolEnv
from carpool_ql import QLearning
import time
import carpool_data as cd
from tqdm import tqdm
import argparse

START_IDX = 1
GOAL_IDX = 0
CAPACITY = 4

UNIT_DIVISOR = 1000.0


def str2bool(v):
    return v.lower() in ('true', '1')


class CarpoolDataset():
    def __init__(self, dataset_fname):
        self.data_set = []
        with open(dataset_fname, 'r') as dset:
            for l in dset:
                x = np.array(l.split(), dtype=np.float32).reshape([-1, 2])
                self.data_set.append(x)

    def __getitem__(self, idx):
        return self.data_set[idx]

    def __len__(self):
        return len(self.data_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Carpooling with RL - Testing")

    parser.add_argument('--test_file', default='test_data/carpool-size-10-len-100-test.txt')
    parser.add_argument('--dist_file', default='map_data/distance_matrix_test.csv')
    parser.add_argument('--coords_file', default='map_data/carpool_map_coordinates_test.csv')
    parser.add_argument('--num_episodes', default=1000)
    parser.add_argument('--lr', default=0.05)
    parser.add_argument('--eps', default=1.0)
    parser.add_argument('--eps_min', default=0.1)
    parser.add_argument('--eps_decay', default=0.00005)
    parser.add_argument('--gamma', default=1.0)
    parser.add_argument('--disable_progress_bar', type=str2bool, default=False)
    parser.add_argument('--seed', default=123456789)

    args = vars(parser.parse_args())

    dataset = CarpoolDataset(args['test_file'])
    data_len = len(dataset)
    mgr = cd.CarpoolDataManager(args['dist_file'], args['coords_file'])

    times = []
    passengers = []
    route_lengths = []

    np.random.seed(args['seed'])
    for i, data in enumerate(tqdm(dataset, disable=args['disable_progress_bar'])):
        dist = mgr.distances_pts(data[1:])
        delay = data[0, 0] * UNIT_DIVISOR - dist[START_IDX, GOAL_IDX]

        env = CarpoolEnv(dist, START_IDX, GOAL_IDX, CAPACITY, delay)
        q_obj = QLearning(env,
                          lr=args['lr'],
                          eps=args['eps'],
                          eps_min=args['eps_min'],
                          eps_decay=args['eps_decay'],
                          gamma=args['gamma'])

        time_start = time.time()
        q_obj.run(args['num_episodes'])
        route, length, reward = q_obj.select_route()
        time_elapsed = time.time() - time_start

        times.append(time_elapsed)
        passengers.append(len(route) - 2)
        route_lengths.append(length)

    print('Avg route length:', np.mean(route_lengths))
    print('Avg passengers picked:', np.mean(passengers))
    print('Time to evaluate:', '%.2f' % np.sum(times), 'sec')
