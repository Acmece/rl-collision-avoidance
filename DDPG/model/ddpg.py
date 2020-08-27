import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import soft_update

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname + 'ddpg'):
    os.makedirs('./log/' + hostname + 'ddpg')
ppo_file = './log/' + hostname + 'ddpg' + '/ddpg.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)

## gennerate action
#------------------------------------------------------------------------------------
def generate_action(env, state_list, actor, action_bound):
    """
        returns mean(sigmoid(v), tanh(w)), r_action(cliped action)
    """
    if env.index == 0:
        s_list, goal_list, speed_list = [], [], []
        for i in state_list:
            s_list.append(i[0])
            goal_list.append(i[1])
            speed_list.append(i[2])

        s_list = np.asarray(s_list)
        goal_list = np.asarray(goal_list)
        speed_list = np.asarray(speed_list)


        s_list = Variable(torch.from_numpy(s_list)).float().cuda()
        goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
        speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

        mean, action = actor(s_list, goal_list, speed_list)
        mean = mean.data.cpu().numpy()
        action = action.data.cpu().numpy()
        r_action = action.clip(action_bound[0], action_bound[1])

    else:
        mean = None
        r_action = None

    return mean, r_action

#------------------------------------------------------------------------------------
## policy update function (ddpg)
#------------------------------------------------------------------------------------
def ddpg_update_stage(policy, optimizer, batch_size, memory, epoch, replay_size, gamma, num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4, tau=0.001):

    actor, actor_target, critic, critic_target = policy
    actor_opt, critic_opt = optimizer
    
    for update in range(epoch):
        
        ## Sample a batch of transitions from replay buffer:
        #----------------------------------------------------------------------------------------------------------
        obss, goals,speeds, actions, rewards, n_obss,n_goals, n_speeds, masks = memory.sample(batch_size)
    
        obss = obss.reshape((num_step*num_env, frames, obs_size))
        goals = goals.reshape((num_step*num_env, 2))
        speeds = speeds.reshape((num_step*num_env, 2))
        
        actions = actions.reshape(num_step*num_env, act_size)

        rewards = rewards.reshape((num_step*num_env, 1))
        
        n_obss = n_obss.reshape((num_step*num_env, frames, obs_size))
        n_goals = n_goals.reshape((num_step*num_env, 2))
        n_speeds = n_speeds.reshape((num_step*num_env, 2))
        
        masks = masks.reshape((num_step*num_env, 1))
        
        sampled_obs = Variable(torch.from_numpy(obss)).float().cuda()
        sampled_goals = Variable(torch.from_numpy(goals)).float().cuda()
        sampled_speeds = Variable(torch.from_numpy(speeds)).float().cuda()

        sampled_n_obs = Variable(torch.from_numpy(n_obss)).float().cuda()
        sampled_n_goals = Variable(torch.from_numpy(n_goals)).float().cuda()
        sampled_n_speeds = Variable(torch.from_numpy(n_speeds)).float().cuda()

        sampled_actions = Variable(torch.from_numpy(actions)).float().cuda()

        sampled_rewards = Variable(torch.from_numpy(rewards)).float().cuda()
        sampled_masks = Variable(torch.from_numpy(masks)).float().cuda()
        

        ## Select next action according to target policy
        #----------------------------------------------------------------------------------------------------------
        sampled_n_actions, _ = actor_target(sampled_n_obs, sampled_n_goals, sampled_n_speeds)
        sampled_n_values = critic_target(sampled_n_obs, sampled_n_goals, sampled_n_speeds, sampled_n_actions)
        expected_values = sampled_rewards + (1.0 - sampled_masks) * gamma*sampled_n_values
        sampled_s_action = critic(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

        ## Optimize critic
        #----------------------------------------------------------------------------------------------------------
        critic_opt.zero_grad()
        loss_value = F.mse_loss(sampled_s_action, expected_values.detach())
        loss_value.backward()
        critic_opt.step()

        actor_action, _ = actor(sampled_obs, sampled_goals, sampled_speeds)

        ## Optimize actor
        #----------------------------------------------------------------------------------------------------------
        actor_opt.zero_grad()
        loss_policy = (-critic(sampled_obs, sampled_goals, sampled_speeds, actor_action)).mean()
        loss_policy.backward()
        actor_opt.step()

        ## Soft update
        #----------------------------------------------------------------------------------------------------------
        soft_update(actor_target, actor, tau)
        soft_update(critic_target, critic, tau)
