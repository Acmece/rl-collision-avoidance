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
ppo_file = './log/' + hostname + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)


def transform_buffer(buff):
    s_batch, goal_batch, speed_batch, a_batch, l_batch, r_batch,\
    s_n_batch, goal_n_batch, speed_n_batch, d_batch = [], [], [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []
    s_n_temp, goal_n_temp, speed_n_temp = [], [], []

    for e in buff:
        for state in e[0]:
 
            s_temp.append(state[0])
            goal_temp.append(state[1])
            speed_temp.append(state[2])

        for state_n in e[4]:
            s_n_temp.append(state_n[0])
            goal_n_temp.append(state_n[1])
            speed_n_temp.append(state_n[2])
        
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        
        s_n_batch.append(s_n_temp)
        goal_n_batch.append(goal_n_temp)
        speed_n_batch.append(speed_n_temp)
        
        s_temp = []
        goal_temp = []
        speed_temp = []
        
        s_n_temp = []
        goal_n_temp = []
        speed_n_temp = []

        a_batch.append(e[1])
        l_batch.append(e[2])
        r_batch.append(e[3])
        d_batch.append(e[5])

    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    l_batch = np.asarray(l_batch)
    r_batch = np.asarray(r_batch)
    s_n_batch = np.asarray(s_n_batch)
    goal_n_batch = np.asarray(goal_n_batch)
    speed_n_batch = np.asarray(speed_n_batch)
    d_batch = np.asarray(d_batch)
    

    return s_batch, goal_batch, speed_batch, a_batch, l_batch, r_batch, s_n_batch, goal_n_batch, speed_n_batch, d_batch


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


def sac_update_stage(policy, optimizer, critic, critic_opt, critic_target, batch_size, memory, epoch,
               replay_size, tau, alpha, gamma, updates, update_interval,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):

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

            n_q_value = sampled_rewards + sampled_masks * gamma * (min_qf_n_target)

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


def ppo_update_stage1(policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    obss = obss.reshape((num_step*num_env, frames, obs_size))
    goals = goals.reshape((num_step*num_env, 2))
    speeds = speeds.reshape((num_step*num_env, 2))
    actions = actions.reshape(num_step*num_env, act_size)
    logprobs = logprobs.reshape(num_step*num_env, 1)
    advs = advs.reshape(num_step*num_env, 1)
    targets = targets.reshape(num_step*num_env, 1)

    for update in range(epoch):
        sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=batch_size,
                               drop_last=False)
        for i, index in enumerate(sampler):
            sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
            sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
            sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

            sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
            sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
            sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
            sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()


            new_value, new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + 20 * value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                     float(value_loss.detach().cpu().numpy()), float(
                                                    dist_entropy.detach().cpu().numpy())
            logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))

    print('update')