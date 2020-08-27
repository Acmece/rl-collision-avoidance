import random
import numpy as np

class ReplayMemory:
'''
    current lidar(t-2, t-1, t)_data, current goal(x,y) data, current speed(v_x, v_y) data, action, reward, next lidar(t-2, t-1, t)_data, next goal(x,y) data, next speed(v_x, v_y) data, terminate data
'''
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, lidar,goal,speed, action, reward, n_lidar, n_goal, n_speed, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (lidar,goal,speed, action, reward, n_lidar, n_goal, n_speed, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        lidar,goal,speed, action, reward, n_lidar,n_goal,n_speed, done = map(np.stack, zip(*batch))
        return lidar,goal,speed, action, reward, n_lidar,n_goal,n_speed, done

    def __len__(self):
        return len(self.buffer)
