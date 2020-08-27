import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from model.net import Policy, Value
from simple_world import StageWorld
from model.trpo import trpo_update_stage, generate_train_data
from model.trpo import generate_action, generate_value
from model.trpo import transform_buffer


MAX_EPISODES = 120000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 300#3072
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 1
OBS_SIZE = 512
ACT_SIZE = 2
P_LEARNING_RATE = 5e-5
V_LEARNING_RATE = 5e-5
L2_RATE = 0.001
MAX_KL = 0.01

def run(comm, env, policy, value, policy_path, action_bound, policy_opt, value_opt):

    # rate = rospy.Rate(5)
    buff = []
    global_update = 0
    global_step = 0

    #world reset
    if env.index == 0:
        env.reset_world()


    for id in range(MAX_EPISODES):
        
        ## reset
        #--------------------------------------------------
        env.reset_pose()

        terminal = False
        ep_reward = 0
        step = 1

        ## generate goal
        #--------------------------------------------------
        env.generate_goal_point()

        ## get_state
        #--------------------------------------------------
        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]

        
        while not terminal and not rospy.is_shutdown():
        
            state_list = comm.gather(state, root=0)

            ## get_action and value
            #-------------------------------------------------------------------------
            # generate actions and value at rank==0
            a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)
            v = generate_value(env=env, state_list=state_list, value=value)


            ## execute actions
            #--------------------------------------------------
            real_action = comm.scatter(scaled_action, root=0)
            
            ## run action
            #--------------------------------------------------
            env.control_vel(real_action)
            
            rospy.sleep(0.001)

            ## get reward
            #-------------------------------------------------------------------------
            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1

        
            ## get next state
            #-------------------------------------------------------------------------

            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)
            #-------------------------------------------------------------------------

            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                last_v = generate_value(env=env, state_list=state_next_list, value=value)


            ## training
            #-------------------------------------------------------------------------
            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:

                    ## memory saver
                    #---------------------------------------------------------------------------------------------------        
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    
                    ## get target & get advantage function
                    #---------------------------------------------------------------------------------------------------        
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    
                    ## update
                    #---------------------------------------------------------------------------------------------------        
                    trpo_update_stage(policy=policy, policy_opt=policy_opt, value=value, value_opt=value_opt,
                                            batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, max_kl=MAX_KL, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []
                    global_update += 1

            step += 1
            state = state_next


        ## save policy
        #-------------------------------------------------------------------------
        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/policy_{}'.format(global_update))
                torch.save(value.state_dict(), policy_path + '/value_{}'.format(global_update))

                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)


if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname + '_trpo'):
        os.makedirs('./log/' + hostname + '_trpo')
    output_file = './log/' + hostname + '_trpo' + '/output.log'
    cal_file = './log/' + hostname + '_trpo' + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    
    print("ENV")
    
    reward = None
    action_bound = [[0, -1], [1, 1]] ####
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy_0819_trpo'
        # policy = MLPPolicy(obs_size, act_size)
        policy = Policy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        value = Value(frames=LASER_HIST, action_space=2)
        value.cuda()
        
        policy_opt = Adam(policy.parameters(), lr=P_LEARNING_RATE)
        value_opt = Adam(value.parameters(), lr=V_LEARNING_RATE, weight_decay=L2_RATE)
        
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        policy_file = policy_path + '/stage1_2.pth'
        if os.path.exists(policy_file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(policy_file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
        
        value_file = policy_path + '/stage1_2.pth'
        if os.path.exists(value_file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(value_file)
            value.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy = None

        policy_path = None
        policy_opt = None
        value_opt = None

    try:
        run(comm=comm, env=env, policy=policy, value=value,  policy_path=policy_path, action_bound=action_bound, policy_opt=policy_opt, value_opt=value_opt)
    except KeyboardInterrupt:
        pass
