import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from net import Policy, Value

hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname + 'reward_g'):
    os.makedirs('./log/' + hostname + 'reward_g')
ppo_file = './log/' + hostname + 'reward_g' + '/trpo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)

## transform_buffer
#-------------------------------------------------------------------------------
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


## generate_action
#-------------------------------------------------------------------------------
def generate_action(env, state_list, policy, action_bound):
"""
    returns action, log_action_prob, scaled_action(cliped action)
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


        a, logprob, mean, _, _ = policy(s_list, goal_list, speed_list)
        a, logprob = a.data.cpu().numpy(), logprob.data.cpu().numpy()
        scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
    else:
        a = None
        scaled_action = None
        logprob = None

    return a, logprob, scaled_action


## generate_value
#-------------------------------------------------------------------------------
def generate_value(env, state_list, value):
"""
    returns value estimate
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

        v = value(s_list, goal_list, speed_list)
        v = v.data.cpu().numpy()

    else:
        v = None
        
    return v


## generate_action for test
#-------------------------------------------------------------------------------
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

        _, _, mean, _, _ = policy(s_list, goal_list, speed_list)
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


def trpo_update_stage(policy, policy_opt, value, value_opt, batch_size, memory, epoch, max_kl = 0.01,
               num_step=2048, num_env=12, frames=1, obs_size=24, act_size=4):

    obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

    
    # ----------------------------
    # step 1: get returns and GAEs
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
            

            # ----------------------------
            # step 3: get gradient of loss and hessian of kl
            sampled_targets_1 = sampled_targets.view(-1, 1)
            sampled_targets_2 = sampled_advs.view(-1, 1)

            new_value = value(sampled_obs, sampled_goals, sampled_speeds)

            value_loss = F.mse_loss(new_value, sampled_targets_1 + sampled_targets_2)
            
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

            #---------------------------------------------------
            new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

            sampled_logprobs = sampled_logprobs.view(-1, 1)
            ratio = torch.exp(new_logprob - sampled_logprobs)

            sampled_advs = sampled_advs.view(-1, 1)
            surrogate = ratio * sampled_advs
            loss = surrogate.mean()
            
            loss_grad = torch.autograd.grad(loss, policy.parameters())
            loss_grad = flat_grad(loss_grad)

            step_dir = conjugate_gradient(policy, sampled_obs, sampled_goals, sampled_speeds, loss_grad.data, nsteps=10)
            

            # ----------------------------
            # step 4: get step direction and step size and full step

            params = flat_params(policy)
            shs = 0.5 * (step_dir.cuda() * fisher_vector_product(policy, sampled_obs, sampled_goals, sampled_speeds, step_dir)).sum(0, keepdim=True)
            step_size = 1 / torch.sqrt(shs / max_kl)[0]
            
            full_step = (step_size * step_dir).cuda()


            # ----------------------------
            # step 5: do backtracking line search for n times


            old_policy = Policy(frames=frames, action_space=2)
            old_policy.cuda()


            old_policy = old_policy #input
            update_model(old_policy, params)
            expected_improve = (loss_grad * full_step).sum(0, keepdim=True)

            flag = False
            fraction = 1.0

            for i in range(10):
                new_params = params + fraction * full_step
                update_model(policy, new_params)

                new_logprob, dist_entropy = policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)

                ratio = torch.exp(new_logprob - sampled_logprobs)

                surrogate = ratio * sampled_advs
                new_loss = surrogate.mean()

                loss_improve = new_loss - loss
                expected_improve *= fraction
                kl = kl_divergence(new_actor=policy, old_actor=old_policy, obss=sampled_obs, goals=sampled_goals, speeds=sampled_speeds)
                kl = kl.mean()

                if kl < max_kl and (loss_improve / expected_improve) > 0.5:
                    flag = True
                    break

                fraction *= 0.5

            if not flag:
                params = flat_params(old_policy)
                update_model(policy, params)
                print('policy update does not impove the surrogate')

    print('update')

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

def fisher_vector_product(actor, obss, goals, speeds, p):
    p.detach()
    kl = kl_divergence(new_actor=actor, old_actor=actor, obss=obss, goals=goals, speeds=speeds)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)  # check kl_grad == 0

    kl_grad_p = (kl_grad * p.cuda()).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p.cuda()

def conjugate_gradient(actor, obss, goals, speeds, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = fisher_vector_product(actor, obss, goals, speeds, p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p.cpu()
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def kl_divergence(new_actor, old_actor, obss, goals, speeds):
    _, _, mu, logstd, std = new_actor(obss, goals, speeds)
    _, _, mu_old, logstd_old, std_old = old_actor(obss, goals, speeds)
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)