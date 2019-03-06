import numpy as np
import bisect
import torch
from torch.autograd import Variable

def test_init_pose(index):
    init_pose_list = [[25.00, 0.00, np.pi], [24.80, 3.13, np.pi*26/25], [24.21, 6.22, np.pi*27/25], [23.24, 9.20, np.pi*28/25],
                 [21.91, 12.04, np.pi*29/25], [20.23, 14.69, np.pi*30/25], [18.22, 17.11, np.pi*31/25], [15.94, 19.26, np.pi*32/25],
                 [13.40, 21.11, np.pi*33/25], [10.64, 22.62, np.pi*34/25],
                 [7.73, 23.78, np.pi*35/25], [4.68, 24.56, np.pi*36/25], [1.57, 24.95, np.pi*37/25], [-1.57, 24.95, np.pi*38/25],
                 [-4.68, 24.56, np.pi*39/25], [-7.73, 23.78, np.pi*40/25],
                 [-10.64, 22.62, np.pi*41/25], [-13.40, 21.11, np.pi*42/25], [-15.94, 19.26, np.pi*43/25], [-18.22, 17.11, np.pi*44/25],
                 [-20.23, 14.69, np.pi*45/25], [-21.91, 12.04, np.pi*46/25],
                 [-23.24, 9.20, np.pi*47/25], [-24.21, 6.22, np.pi*48/25], [-24.80, 3.13, np.pi*49/25],[-25.00, -0.00, np.pi*50/25],
                 [-24.80, -3.13, np.pi*51/25], [-24.21, -6.22, np.pi*52/25], [-23.24, -9.20, np.pi*53/25],
                 [-21.91, -12.04, np.pi*54/25], [-20.23, -14.69, np.pi*55/25],
                 [-18.22, -17.11, np.pi*56/25], [-15.94, -19.26, np.pi*57/25], [-13.40, -21.11, np.pi*58/25],
                 [-10.64, -22.62, np.pi*59/25], [-7.73, -23.78, np.pi*60/25],
                 [-4.68, -24.56, np.pi*61/25], [-1.57, -24.95, np.pi*62/25], [1.57, -24.95, np.pi*63/25],
                 [4.68, -24.56, np.pi*64/25], [7.73, -23.78, np.pi*65/25], [10.64, -22.62, np.pi*66/25],
                 [13.40, -21.11, np.pi*67/25], [15.94, -19.26, np.pi*68/25], [18.22, -17.11, np.pi*69/25],
                 [20.23, -14.69, np.pi*70/25], [21.91, -12.04, np.pi*71/25], [23.24, -9.20, np.pi*72/25],
                 [24.21, -6.22, np.pi*73/25], [24.80, -3.13, np.pi*74/25]
                      ]
    return init_pose_list[index]

def test_goal_point(index):
    goal_list = [[-25.00, -0.00], [-24.80, -3.13], [-24.21, -6.22], [-23.24, -9.20], [-21.91, -12.04], [-20.23, -14.69],
                 [-18.22, -17.11], [-15.94, -19.26], [-13.40, -21.11], [-10.64, -22.62], [-7.73, -23.78],
                 [-4.68, -24.56], [-1.57, -24.95], [1.57, -24.95], [4.68, -24.56], [7.73, -23.78], [10.64, -22.62],
                 [13.40, -21.11], [15.94, -19.26], [18.22, -17.11], [20.23, -14.69], [21.91, -12.04], [23.24, -9.20],
                 [24.21, -6.22], [24.80, -3.13], [25.00, 0.00], [24.80, 3.13], [24.21, 6.22], [23.24, 9.20],
                 [21.91, 12.04], [20.23, 14.69], [18.22, 17.11], [15.94, 19.26], [13.40, 21.11], [10.64, 22.62],
                 [7.73, 23.78], [4.68, 24.56], [1.57, 24.95], [-1.57, 24.95], [-4.68, 24.56], [-7.73, 23.78],
                 [-10.64, 22.62], [-13.40, 21.11], [-15.94, 19.26], [-18.22, 17.11], [-20.23, 14.69], [-21.91, 12.04],
                 [-23.24, 9.20], [-24.21, 6.22], [-24.80, 3.13]
                 ]
    return goal_list[index]


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
