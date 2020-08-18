import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname + 'reward_g'):
    os.makedirs('./log/' + hostname + 'reward_g')
ppo_file = './log/' + hostname + 'reward_g' + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


def transform_buffer(buff):
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch = [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []

    for e in buff:
        for state in e[0]:

            s_temp.append(state[0])
            goal_temp.append(state[1])
            speed_temp.append(state[2])

        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        s_temp = []
        goal_temp = []
        speed_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch


def generate_action(env, state_list, actor, action_bound):
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


        a, logprob, mean = actor(s_list, goal_list, speed_list)
        a, logprob = a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
    else:
        a = None
        scaled_action = None
        logprob = None

    return a, logprob, scaled_action



def select_action(env, state_list, actor, action_bound):
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


        a, logprob, mean = actor(s_list, goal_list, speed_list)
        a, logprob, mean = a.data.cpu().numpy(), logprob.data.cpu().numpy(), mean.data.cpu().numpy()
        
    else:
        a = None
        scaled_action = None
        logprob = None
        mean = None

    return mean, logprob


def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    returns = np.zeros((num_step + 1, num_env))
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(num_step)):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


def generate_train_data(rewards, gamma, values, last_value, dones, lam):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    values = list(values)
    values.append(last_value)
    values = np.asarray(values).reshape((num_step+1,num_env))

    targets = np.zeros((num_step, num_env))
    gae = np.zeros((num_env,))

    for t in range(num_step - 1, -1, -1):
        delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
        gae = delta + gamma * lam * (1 - dones[t, :]) * gae

        targets[t, :] = gae + values[t, :]

    advs = targets - values[:-1, :]
    return targets, advs


def td3_update_stage(policy, optimizer, batch_size, memory, epoch, replay_size, gamma, num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4, polyak=0.995, policy_noise=0.2, noise_clip=0.5, policy_delay=2, max_action=1.0, action_bound= [[0, -1], [1, 1]] ):

    actor, actor_target, critic_1, critic_1_target, critic_2, critic_2_target = policy
    actor_opt, critic_1_opt, critic_2_opt = optimizer
    
    for update in range(epoch):
        
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
             
        # Select next action according to target policy
        sampled_noise = sampled_actions.data.normal_(0, policy_noise).cuda()
  
        sampled_noise = sampled_noise.clamp(-noise_clip, noise_clip)

        _ ,_ ,sampled_n_action = (actor_target(sampled_n_obs, sampled_n_goals, sampled_n_speeds))

        sampled_n_action = sampled_n_action + sampled_noise
        
        #???? clip vs clamp (action_bound[])
        sampled_n_action = sampled_n_action.clamp(-max_action, max_action)
       
                
        # Compute target Q-value:
        target_Q1 = critic_1_target(sampled_n_obs, sampled_n_goals, sampled_n_speeds, sampled_n_action)
        target_Q2 = critic_2_target(sampled_n_obs, sampled_n_goals, sampled_n_speeds, sampled_n_action)

        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = sampled_rewards + ((1 - sampled_masks) * gamma * target_Q).detach()


        # Optimize Critic 1:
        current_Q1 = critic_1(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        critic_1_opt.zero_grad()
        loss_Q1.backward()
        critic_1_opt.step()


        # Optimize Critic 2:
        current_Q2 = critic_2(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        critic_2_opt.zero_grad()
        loss_Q2.backward()
        critic_2_opt.step()
        

        # Delayed policy updates:
        if update % policy_delay == 0:
            # Compute actor loss:
            action_data, _, _ = actor(sampled_obs, sampled_goals, sampled_speeds)
            actor_loss = -critic_1(sampled_obs, sampled_goals, sampled_speeds, action_data ).mean()
            

            # Optimize the actor
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            
            # Polyak averaging update:
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
            
            for param, target_param in zip(critic_1.parameters(), critic_1_target.parameters()):
                target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
            
            for param, target_param in zip(critic_2.parameters(), critic_2_target.parameters()):
                target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))    

