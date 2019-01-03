import os
import copy
import numpy as np
import rospy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.autograd import Variable


from model.net import MLPPolicy, CNNPolicy
from stage_world import StageWorld
from model.ppo import ppo_update, generate_train_data

env_name = 'Stage'
env_num = 1
coeff_entropy = 1e-4
lr = 2e-4
mini_batch_size = 512
horizon = 2048
epoch = 1
num_train = 4000
clip_value = 0.2
stack_frame = 3
max_step_per_episode = 10000
is_render = False


MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 100
GAMMA = 0.99
LAMDA = 0.99
BATCH_SIZE = 50
EPOCH = 2
COEFF_ENTROPY = 1e-4
CLIP_VALUE = 0.2
NUM_ENV = 1
OBS_SIZE = 512
ACT_SIZE = 2





def run( env, policy, policy_path, action_bound, optimizer):

    rate = rospy.Rate(5)
    s1 = env.get_laser_observation()
    s_1 = np.stack((s1, s1, s1), axis=1)

    buff = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(env.map, aspect='auto', cmap='hot', vmin=0., vmax=1.5)
    plt.show(block=False)

    for i in range(MAX_EPISODES):
        env.reset_world()
        env.generate_goal_point()
        print('Target: (%.4f, %.4f)' % (env.goal_point[0], env.goal_point[1]))
        terminal = False
        ep_reward = 0
        j = 0

        while not terminal and not rospy.is_shutdown():
            s1 = env.get_laser_observation()
            s_1 = np.append(np.reshape(s1, (LASER_BEAM, 1)), s_1[:, :(LASER_HIST - 1)], axis=1)
            s__1 = np.reshape(s_1, (LASER_HIST, LASER_BEAM))

            goal1 = np.asarray(env.get_local_goal())
            speed1 = np.asarray(env.get_self_speed())

            s__1 = Variable(torch.from_numpy(s__1[np.newaxis])).float().cuda()
            goal1 = Variable(torch.from_numpy(goal1[np.newaxis])).float().cuda()
            speed1 = Variable(torch.from_numpy(speed1[np.newaxis])).float().cuda()
            state1 = [s__1, goal1, speed1]

            map_img = env.render_map([[0, 0], env.goal_point])

            r, terminal, result = env.get_reward_and_terminate(j)
            ep_reward += r

            if j > 0:
                buff.append((state, a[0], r, state1, terminal, logprob, v))
            j += 1
            state = state1

            v, a, logprob, mean = policy(s__1, goal1, speed1)
            v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
            # a = np.asarray([0.4, np.pi/7])
            scaled_action = np.clip(a[0], a_min=action_bound[0], a_max=action_bound[1])
            # print scaled_action
            env.control(scaled_action)

            # plot
            if j == 1:
                im.set_array(map_img)
                fig.canvas.draw()
            # v:(1,1)   a:(1,2)   logprob:(1,1)  s__1:(1,3,512)  goal1:(1,2)  speed1:(1,2)
            if len(buff) > HORIZON-1:
                # state_batch = [e[0] for e in buff]  # attention : list
                # a_batch = np.asarray(e[1] for e in buff)
                # r_batch = np.asarray(e[2] for e in buff)
                # state1_batch = [e[3] for e in buff]  # attention : list
                # d_batch = np.asarray(e[4] for e in buff)
                # l_batch = np.asarray(e[5] for e in buff)
                # v_batch = np.asarray(e[6] for e in buff)
                s_batch, goal_batch, speed_batch, a_batch, r_batch, state1_batch, d_batch, l_batch, \
                v_batch = [], [], [], [], [], [], [], [], []
                for e in buff:
                    s_batch.append(e[0][0].data.cpu().numpy())
                    goal_batch.append(e[0][1].data.cpu().numpy())
                    speed_batch.append(e[0][2].data.cpu().numpy())

                    a_batch.append(e[1])
                    r_batch.append(e[2])
                    state1_batch.append(e[3])
                    d_batch.append(e[4])
                    l_batch.append(e[5])
                    v_batch.append(e[6])

                s_batch = np.asarray(s_batch) # horizon * 1 * frames * obs_size
                goal_batch = np.asarray(goal_batch) # horizon * 1 * 2
                speed_batch = np.asarray(speed_batch) # horiozn * 1 * 2 before reshape
                a_batch = np.asarray(a_batch) # horizon * act_size
                r_batch = np.asarray(r_batch) # horizon
                d_batch = np.asarray(d_batch) # horizon
                l_batch = np.asarray(l_batch) # horizon * 1 * 1
                v_batch = np.asarray(v_batch) # horizon * 1 * 1


                # print s_batch.shape
                # print goal_batch.shape
                # print speed_batch.shape
                # print r_batch.shape
                # print d_batch.shape
                # print l_batch.shape
                # print v_batch.shape


                t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=0.99, values=v_batch,
                                                          last_value=v, dones=d_batch, lam=LAMDA)
                # t_batch : horizon   advs_batch : horiozn

                memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)

                ppo_update(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                           epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                           num_env=NUM_ENV, frames=LASER_HIST,
                           obs_size=OBS_SIZE, act_size=ACT_SIZE)
                buff = []

            rate.sleep()

        # print '| Reward: %.2f' % ep_reward, " | Episode:", i, \
        #     '| Qmax: %.4f' % (ep_ave_max_q / float(j)), \
        #     " | LoopTime: %.4f" % (np.mean(loop_time_buf)), " | Step:", j - 1, '\n'
        print 'Reward: %.2f ' % ep_reward, 'Episode: ', i
        if i % 100 == 0:
            torch.save(policy.state_dict(), policy_path + '/policy.pth')





if __name__ == '__main__':
    # torch.manual_seed(1)
    # np.random.seed(1)
    policy_path = 'policy'
    # policy = MLPPolicy(obs_size, act_size)
    policy = CNNPolicy(frames=stack_frame, action_space=2)
    policy.cuda()
    opt = Adam(policy.parameters(), lr=lr)
    mse = nn.MSELoss()

    if not os.path.exists(policy_path):
        os.makedirs(policy_path)

    file = policy_path + '/policy.pth'
    if os.path.exists(file):
        state_dict = torch.load(file)
        policy.load_state_dict(state_dict)

    env = StageWorld(512)
    reward = None
    action_bound = [[0, -np.pi / 3], [1.2, np.pi / 3]]

    try:
        run(env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass
