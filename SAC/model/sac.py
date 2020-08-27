import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from update_file import soft_update
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/sac.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)

def generate_action(env, state_list, policy, action_bound):
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

        a, logprob, mean = policy(s_list, goal_list, speed_list)
        a, logprob = a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
    else:
        a = None
        scaled_action = None
        logprob = None

    return a, logprob, scaled_action

def generate_action_no_sampling(env, state_list, policy, action_bound):
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

        _, _, mean = policy(s_list, goal_list, speed_list)
        mean = mean.data.cpu().numpy()
        scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
    else:
        mean = None
        scaled_action = None

    return mean, scaled_action

def sac_update_stage(policy, optimizer, critic, critic_opt, critic_target, batch_size, memory, epoch,
               replay_size, tau, alpha, gamma, updates, update_interval,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):
    
    # Sample a batch of transitions from replay buffer:
    obss, goals,speeds, actions, logprobs, rewards, n_obss,n_goals, n_speeds, masks = memory.sample(batch_size)
    
    obss = obss.reshape((num_step*num_env, frames, obs_size))
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    rewards = rewards.reshape((num_step*num_env, 1))
    n_obss = n_obss.reshape((num_step*num_env, frames, obs_size))
    n_goals = n_goals.reshape((num_step*num_env, 2))
    n_speeds = n_speeds.reshape((num_step*num_env, 2))
    masks = masks.reshape((num_step*num_env, 1))
    
    for update in range(epoch):

        sampled_obs = Variable(torch.from_numpy(obss)).float().cuda()
        sampled_goals = Variable(torch.from_numpy(goals)).float().cuda()
        sampled_speeds = Variable(torch.from_numpy(speeds)).float().cuda()

        sampled_n_obs = Variable(torch.from_numpy(n_obss)).float().cuda()
        sampled_n_goals = Variable(torch.from_numpy(n_goals)).float().cuda()
        sampled_n_speeds = Variable(torch.from_numpy(n_speeds)).float().cuda()

        sampled_actions = Variable(torch.from_numpy(actions)).float().cuda()
        sampled_logprobs = Variable(torch.from_numpy(logprobs)).float().cuda()

        sampled_rewards = Variable(torch.from_numpy(rewards)).float().cuda()
        sampled_masks = Variable(torch.from_numpy(masks)).float().cuda()
       

        with torch.no_grad():

            n_actions, n_logprobs, _ = policy(sampled_n_obs, sampled_n_goals, sampled_n_speeds)
            qf1_n_target, qf2_n_target = critic_target(sampled_n_obs, sampled_n_goals, sampled_n_speeds, n_actions)
            min_qf_n_target = torch.min(qf1_n_target, qf2_n_target) - alpha * n_logprobs
            n_q_value = sampled_rewards + (1 - sampled_masks) * gamma * (min_qf_n_target)

        qf1, qf2 = critic(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

        qf1_loss = F.mse_loss(qf1, n_q_value)
        qf2_loss = F.mse_loss(qf2, n_q_value)
        
        qf_loss = qf1_loss + qf2_loss

        critic_opt.zero_grad()
        qf_loss.backward()
        critic_opt.step()
        
        act, log_pi, _ = policy(sampled_obs, sampled_goals, sampled_speeds)

        qf1_pi, qf2_pi= critic(sampled_obs, sampled_goals, sampled_speeds, act)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        alpha_loss = torch.tensor(0.).cuda()
        alpha_tlogs = torch.tensor(alpha) # For TensorboardX logs

        if updates % update_interval == 0:
            soft_update(critic_target, critic, tau)

        updates = updates + 1

    return updates
