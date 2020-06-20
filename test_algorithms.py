import numpy as np
from carpool_env import CarpoolEnv
from carpool_ql import QLearning
import time
import carpool_data as cd

if __name__ == '__main__':

    # sample points and distances
    num_points = 20
    mgr = cd.CarpoolDataManager('map_data/distance_matrix.csv', 'map_data/carpool_map_coordinates.csv')
    mgr.filter_data(max_range=2000, idx_goal=0)
    coords_sample, ids_sample, dist_sample = mgr.sample_data(num_points, idx_goal=0, seed=1)

    # carpool parameters
    start = 7
    goal = 0
    capacity = 4
    delay = 500.0

    # initialize environment
    env = CarpoolEnv(dist_sample, start, goal, capacity, delay)

    # learn Q table
    q_obj = QLearning(env, lr=0.05, eps=1.0, eps_min=0.1, eps_decay=0.00005, gamma=1.0)

    start_time = time.time()
    q_obj.run(num_episodes=5000)
    elapsed_time = time.time() - start_time

    # select optimal route according to Q table
    route, distance, reward = q_obj.select_route()

    print('--------------------------------------')
    print('Recommended route: ', route)
    print('Maximum allowed delay: ', '%.2f' % delay)
    print('Reward: ', '%.2f' % reward)
    print('Distance: ', distance)
    print('Convergence time: ', '%.2f' % elapsed_time, 'sec')
    print('--------------------------------------')

    q_obj.plot()

    # write carpool_solution to file
    ids = np.arange(num_points)
    in_route = np.zeros(len(ids_sample), dtype=np.bool)
    in_route[np.asarray(route)] = True

    cd.save_coordinates('route.csv', coords_sample, ids=ids, extras=in_route)