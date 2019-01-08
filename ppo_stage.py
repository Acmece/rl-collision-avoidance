import os
import copy
import numpy as np
import rospy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpi4py import MPI

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
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.99
BATCH_SIZE = 64
EPOCH = 2
COEFF_ENTROPY = 1e-4
CLIP_VALUE = 0.2
NUM_ENV = 12
OBS_SIZE = 512
ACT_SIZE = 2





def run(comm, env, policy, policy_path, action_bound, optimizer):

    rate = rospy.Rate(5)
    s1 = env.get_laser_observation()
    s_1 = np.stack((s1, s1, s1), axis=1)

    buff = []

    if env.index == 0:
        env.reset_world()
    env.generate_goal_point()


    for id in range(MAX_EPISODES):
        # print 'Goal: (%.4f, %.4f)' % (env.goal_point[0], env.goal_point[1])
        terminal = False
        ep_reward = 0
        j = 0


        while not terminal and not rospy.is_shutdown():
            s1 = env.get_laser_observation()
            s_1 = np.append(np.reshape(s1, (LASER_BEAM, 1)), s_1[:, :(LASER_HIST - 1)], axis=1)
            s__1 = np.reshape(s_1, (LASER_HIST, LASER_BEAM))

            goal1 = np.asarray(env.get_local_goal())
            speed1 = np.asarray(env.get_self_speed())

            state1 = [s__1, goal1, speed1]



            r, terminal, result = env.get_reward_and_terminate(j)
            ep_reward += r

            # for MPI
            state1_list = comm.gather(state1, root=0)
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)


            if j > 0 and env.index == 0:
                if env.index == 0:
                    buff.append((state_list, scaled_action, r_list, state1_list, terminal_list, logprob, v))
            j += 1
            state_list = state1_list

            if env.index == 0:
                s_list, goal_list, speed_list = [], [], []
                for i in state_list:
                    s_list.append(i[0])
                    goal_list.append(i[1])
                    speed_list.append(i[2])

                s_list = np.asarray(s_list)
                goal_list = np.asarray(goal_list)
                speed_list = np.asarray(speed_list)

                # print s_list.shape
                # print goal_list.shape
                # print speed_list.shape


                s_list = Variable(torch.from_numpy(s_list)).float().cuda()
                goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
                speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
                # print s_list.shape
                # print goal_list.shape
                # print speed_list.shape
                # exit()

                v, a, logprob, mean = policy(s_list, goal_list, speed_list)
                v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
                scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
            else:
                v = None
                scaled_action = None
                logprob = None


            real_action = comm.scatter(scaled_action, root=0)

            env.control_vel(real_action)


            # v:(12,1)   a:(12,2)   logprob:(12,1)  s__1:(12,3,512)  goal1:(12,2)  speed1:(12,2)
            if len(buff) > HORIZON-1 and env.index == 0:
                s_batch, goal_batch, speed_batch, a_batch, r_batch, state1_batch, d_batch, l_batch, \
                v_batch = [], [], [], [], [], [], [], [], []
                s_temp = []
                goal_temp = []
                speed_temp = []


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
                    state1_batch.append(e[3])
                    d_batch.append(e[4])
                    l_batch.append(e[5])
                    v_batch.append(e[6])

                s_batch = np.asarray(s_batch) # horizon * 12 * frames * obs_size
                goal_batch = np.asarray(goal_batch) # horizon * 12 * 2
                speed_batch = np.asarray(speed_batch) # horiozn * 12 * 2
                a_batch = np.asarray(a_batch) # horizon * 12 * act_size
                r_batch = np.asarray(r_batch) # horizon * 12
                d_batch = np.asarray(d_batch) # horizon * 12
                l_batch = np.asarray(l_batch) # horizon * 12 * 1
                v_batch = np.asarray(v_batch) # horizon * 12 * 1

                # print s_batch.shape
                # print goal_batch.shape
                # print speed_batch.shape
                # print a_batch.shape
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





        print 'Env {}, Episode {}, Reward {}, finished'.format(env.index, id, ep_reward)

        if id % 10 == 0 and env.index == 0:
            torch.save(policy.state_dict(), policy_path + '/policy.pth')





if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = StageWorld(512, index=rank)
    reward = None
    action_bound = [[0, -np.pi / 3], [1.2, np.pi / 3]]

    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
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
            print 'load model'
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
    else:
        policy = None
        policy_path = None
        opt = None


    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass
