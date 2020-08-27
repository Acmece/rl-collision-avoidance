import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from model.utils import log_normal_density

class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.shape[0], 1,  -1)

#--------------------------------------------------------------------------------------------------------
## Critic
#--------------------------------------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, frames, action_space):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2+2, 128)
        self.critic1 = nn.Linear(128, 1)

        # Q2 architecture
        self.crt_fea_cv3 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc3 = nn.Linear(128*32, 256)
        self.crt_fc4 = nn.Linear(256+2+2+2, 128)
        self.critic2 = nn.Linear(128, 1)

    def forward(self, x, goal, speed, action):
        """
            returns value1 estimation, value2 estimation
        """
        
        v1 = F.relu(self.crt_fea_cv1(x))
        v1 = F.relu(self.crt_fea_cv2(v1))
        v1 = v1.view(v1.shape[0], -1)
        v1 = F.relu(self.crt_fc1(v1))
        v1 = torch.cat((v1, goal, speed, action), dim=-1)
        v1 = F.relu(self.crt_fc2(v1))
        v1 = self.critic1(v1)

        v2 = F.relu(self.crt_fea_cv1(x))
        v2 = F.relu(self.crt_fea_cv2(v2))
        v2 = v2.view(v2.shape[0], -1)
        v2 = F.relu(self.crt_fc1(v2))
        v2 = torch.cat((v2, goal, speed, action), dim=-1)
        v2 = F.relu(self.crt_fc2(v2))
        v2 = self.critic2(v2)

        return v1, v2

#--------------------------------------------------------------------------------------------------------
## Actor
#--------------------------------------------------------------------------------------------------------
class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space):
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 =  nn.Linear(256+2+2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

    def forward(self, x, goal, speed):
        """
            returns action, log_action_prob, mean(sigmoid, tanh)
        """
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))

        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)
        
        return action, logprob, mean

    def evaluate_actions(self, x, goal, speed, action):
        """
            returns log_action_prob, distance entropy
        """
        _, _, mean = self.forward(x, goal, speed)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return logprob, dist_entropy



if __name__ == '__main__':
    from torch.autograd import Variable


