import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def calculate_returns(rewards, dones, last_value, values, gamma=0.99):
    num_step = rewards.shape[0]
    num_env = rewards.shape[1]
    returns = np.zeros((num_step + 1, num_env))
    returns[-1] = last_value
    dones = 1 - dones
    for i in reversed(range(num_step)):
        returns[i] = gamma * returns[i+1] * dones[i] + rewards[i]
    return returns


# def generate_train_data(rewards, gamma, values, last_value, dones, lam):
#     num_step = rewards.shape[0]
#     num_env = rewards.shape[1]
#     values = list(values)
#     values.append(last_value)
#     values = np.asarray(values).reshape((-1, num_env))
#
#     targets = np.zeros((num_step, num_env))
#     gae = np.zeros_like([num_env, ])
#     for t in range(num_step - 1, -1, -1):
#         delta = rewards[t, :] + gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
#         gae = delta + gamma * lam * (1 - dones[t, :]) * gae
#
#         targets[t, :] = gae + values[t, :]
#
#     advs = targets - values[:-1, :]
#
#     return targets, advs

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

    # print advs.shape  # horizon * 12
    # print targets.shape # horizon * 12

    return targets, advs



def ppo_update(policy, optimizer, batch_size, memory, epoch,
               coeff_entropy=0.02, clip_value=0.2,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):
    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    advs = (advs - advs.mean()) / advs.std()

    # print(targets.shape) # horizon * 12
    # print(values.shape) # horizon * 12 * 1
    # print(actions.shape) # horizon * 12 * act_size
    # print(logprobs.shape) # horizon * 12 * 1
    # print(advs.shape) # horizon * 12

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

            # print(dist_entropy.shape) # schalor


            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate1 = ratio * sampled_advs
            surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            sampled_targets = sampled_targets.view(-1, 1)
            value_loss = F.mse_loss(new_value, sampled_targets)

            loss = policy_loss + value_loss - coeff_entropy * dist_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('update')
    # print(value_loss.data, policy_loss.data, dist_entropy.data)
    # return value_loss.data[0], policy_loss.data[0], dist_entropy.data[0]



def generate_trajectory(workers, policy, max_step, agent_conns, obs_size=None,
                        num_env=None, stack_frame=None,
                        gamma=0.99, lam=0.99):
    """generate a batch of examples using policy"""

    nstep = 0
    # obs = env.reset()
    # obss = np.zeros([num_env, stack_frame, obs_size])
    obss = []
    targets = []
    speeds = []
    for work in workers:
        obs, target, speed = work.first_state
        obss.append(obs)
        targets.append(target)
        speeds.append(speed)

    obss = np.asarray(obss)
    goals = np.asarray(targets)
    speeds = np.asarray(speeds)

    done = False
    total_obss, total_rewards, total_actions, total_logprobs, total_dones, \
    total_values, total_goals, total_speeds = [], [], [], [], [], [], [], []
    while not (nstep == max_step):
        # if is_render:
        #     env.render()

        obss = Variable(torch.from_numpy(obss[np.newaxis])).float().cuda()
        goals = Variable(torch.from_numpy(goals[np.newaxis])).float().cuda()
        speeds = Variable(torch.from_numpy(speeds[np.newaxis])).float().cuda()

        obss = obss.view(-1, stack_frame, obs_size)
        goals = goals.view(-1, 1, 2)
        speeds = speeds.view(-1, 1,  2)

        value, action, logprob, mean = policy(obss, goals, speeds)
        value, action, logprob = value.data.cpu().numpy(), action.data.cpu().numpy(), \
                                 logprob.data.cpu().numpy()


        # stack the result of env
        for agent_conn, action_ in zip(agent_conns, action):
            agent_conn.send(action_)


        next_obss, next_goals, next_speeds, rewards, dones= [], [], [], [], []
        for agent_conn in agent_conns:
            state, reward, terminal, _ = agent_conn.recv()
            obs, goal, speed = state
            next_obss.append(obs)
            next_goals.append(goal)
            next_speeds.append(speed)
            dones.append(terminal)
            rewards.append(reward)
#   need to do things here
        total_actions.append(action)
        total_values.append(value)
        total_logprobs.append(logprob)
        total_rewards.append(rewards)
        total_dones.append(dones)

        total_obss.append(obss.data.cpu().numpy())
        total_goals.append(goals.data.cpu().numpy())
        total_speeds.append(speeds.data.cpu().numpy())

        obss = np.stack(next_obss)
        goals = np.stack(next_goals)
        speeds = np.stack(next_speeds)

        nstep += 1

    obss = Variable(torch.from_numpy(obss[np.newaxis])).float().cuda()
    goals = Variable(torch.from_numpy(goals[np.newaxis])).float().cuda()
    speeds = Variable(torch.from_numpy(speeds[np.newaxis])).float().cuda()

    obss = obss.view(-1, stack_frame, obs_size)
    targets = targets.view(-1, 1, 2)
    speeds = speeds.view(-1, 1, 2)

    assert obss.shape == (num_env, stack_frame, 24)
    assert goals.shape == (num_env, 2)
    assert speeds.shape == (num_env, 2)

    value, _, _, _ = policy(obss, goals, speeds)
    last_value = value.data.cpu().numpy()

    total_obss = np.asarray(total_obss)
    total_goals = np.asarray(total_goals)
    total_speeds = np.asarray(total_speeds)

    total_rewards = np.asarray(total_rewards)
    total_logprobs = np.asarray(total_logprobs)
    total_dones = np.asarray(total_dones)
    total_values = np.asarray(total_values)
    total_actions = np.asarray(total_actions)

    # observations = np.asarray(observations)
    # rewards = np.asarray(rewards)
    # logprobs = np.asarray(logprobs)
    # dones = np.asarray(dones)
    # values = np.asarray(values)
    # actions = np.asarray(actions)

    # print('total_rewards {} '.format(total_rewards.shape))
    # print('total_dones {} '.format(total_dones.shape))
    # print('total_values {} '.format(total_values.shape))
    # print('last value {} '.format(last_value.shape))

    # total_returns = calculate_returns(total_rewards, total_dones, last_value, total_values) # (num_step+1) * num_env
    # print('total returns {} '.format(total_returns.shape))
    total_target, total_adv = generate_train_data(rewards=total_rewards, values=total_values,
                                                  last_value=last_value, dones=total_dones,
                                                  gamma=gamma, lam=lam)


    # return total_obss, total_actions, total_logprobs, total_returns, total_values, total_rewards
    return [total_obss, total_goals, total_speeds], total_actions, total_logprobs, total_target, total_values, total_rewards, total_adv


