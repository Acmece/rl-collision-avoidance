import numpy as np
import bisect
import torch
from torch.autograd import Variable


def get_init_pose(index):
    init_pose_list = [[-7.00, 11.50, np.pi], [-7.00, 9.50, np.pi], [-18.00, 11.50, 0.00], [-18.00, 9.50, 0.00],
                      [-12.50, 17.00, np.pi*3/2], [-12.50, 4.00, np.pi/2], [-2.00, 16.00, -np.pi/2], [0.00, 16.00, -np.pi/2],
                      [3.00, 16.00, -np.pi/2], [5.00, 16.00, -np.pi/2], [10.00, 4.00, np.pi/2], [12.00, 4.00, np.pi/2],
                      [14.00, 4.00, np.pi/2], [16.00, 4.00, np.pi/2], [18.00, 4.00, np.pi/2], [-2.5, -2.5, 0.00],
                      [-0.5, -2.5, 0.00], [3.5, -2.5, np.pi], [5.5, -2.5, np.pi], [-2.5, -18.5, np.pi/2],
                      [-0.5, -18.5, np.pi/2], [1.5, -18.5, np.pi/2], [3.5, -18.5, np.pi/2], [5.5, -18.5, np.pi/2],
                      [-6.00, -10.00, np.pi], [-7.15, -6.47, np.pi*6/5], [-10.15, -4.29, np.pi*7/5], [-13.85, -4.29, np.pi*8/5],
                      [-16.85, -6.47, np.pi*9/5], [-18.00, -10.00, np.pi*2], [-16.85, -13.53, np.pi*11/5], [-13.85, -15.71, np.pi*12/5],
                      [-10.15, -15.71, np.pi*13/5], [-7.15, -13.53, np.pi*14/5], [10.00, -17.00, np.pi/2], [12.00, -17.00, np.pi/2],
                      [14.00, -17.00, np.pi/2], [16.00, -17.00, np.pi/2], [18.00, -17.00, np.pi/2], [10.00, -2.00, -np.pi/2],
                      [12.00, -2.00, -np.pi/2], [14.00, -2.00, -np.pi/2], [16.00, -2.00, -np.pi/2], [18.00, -2.00, -np.pi/2]]
    return init_pose_list[index]

def get_goal_point(index):
    goal_list = [[-18.0, 11.5], [-18.0, 9.5], [-7.0, 11.5], [-7.0, 9.5], [-12.5, 4.0], [-12.5, 17.0],
                 [-2.0, 3.0], [0.0, 3.0], [3.0, 3.0], [5.0, 3.0], [10.0, 10.0], [12.0, 10.0],
                 [14.0, 10.0], [16.0, 10.0], [18.0, 10.0], [3.5, -2.5], [5.5, -2.5], [-2.5, -2.5],
                 [-0.5, -2.5], [-2.5, -5.5], [-0.5, -5.5], [1.5, -5.5], [3.5, -5.5], [5.5, -5.5],
                 [-18.0, -10.0], [-16.85, -13.53], [-13.85, -15.71], [-10.15, -15.71], [-7.15, -13.53], [-6.00, -10.00],
                 [-7.15, -6.47], [-10.15, -4.29], [-13.85, -4.29], [-16.85, -6.47],
                 ]
    return goal_list[index]

def get_filter_index(d_list):
    filter_index = []
    filter_flag = 0
    step = d_list.shape[0]
    num_env = d_list.shape[1]
    for i in range(num_env):
        for j in range(step):
            if d_list[j, i] == True:
                filter_flag += 1
            else:
                filter_flag = 0
            if filter_flag >= 2:
                filter_index.append(num_env*j + i)
    return filter_index


def get_group_terminal(terminal_list, index):
    group_terminal = False
    refer = [0, 6, 10, 15, 19, 24, 34, 44]
    r = bisect.bisect(refer, index)
    if reduce(lambda x, y: x * y, terminal_list[refer[r-1]:refer[r]]) == 1:
        group_terminal = True
    return group_terminal


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""

    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True) # num_env * frames * 1
    return log_density



class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#algorithm_Parallel
    def __init__(self, epsilon=1e-4, shape=()):  # shape (1, 1, 84, 84)
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
