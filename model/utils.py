import numpy as np
from gym.wrappers import Monitor
import torch
from torch.autograd import Variable


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""


    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 *\
        np.log(2 * np.pi) - log_std    # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True) # num_env * frames * 1
    return log_density


def enjoy(policy, env, save_path=None, save_video=False, obs_fn=None,
          nepochs=100):
    """        Enjoy and flush your result using Monitor class.
    """
    if save_video:
        assert save_path is not None, 'A path to save videos must be provided!'
    policy.cuda()
    policy.eval()
    if save_video:
        env = Monitor(env, directory=save_path)

    for e in range(0, 100):
        done = False
        obs = env.reset()
        episode_rwd = 0
        while not done:
            env.render()
            if obs_fn is not None:
                obs = obs_fn(obs)
            obs = Variable(torch.from_numpy(obs[np.newaxis])).float().cuda()
            value, action, logprob, mean = policy(obs)
            action = action.data[0].cpu().numpy()
            obs, reward, done, _ = env.step(action)
            episode_rwd += reward
        print('Episode reward is', episode_rwd)


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
