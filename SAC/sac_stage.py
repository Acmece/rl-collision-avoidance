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

from model.net import QNetwork, CNNPolicy
from stage_world import StageWorld
from model.sac import sac_update_stage
from model.sac import generate_action
from model.update_file import hard_update
from model.replay_memory import ReplayMemory


MAX_EPISODES = 100000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 3072
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 1
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 1
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5

REPLAY_SIZE = 100000
TAU = 0.005
UPDATE_INTERVAL = 1
ALPHA = 0.2
SEED = 123456

def run(comm, env, policy, critic, critic_opt, critic_target, policy_path, action_bound, optimizer):

    buff = []
    global_update = 0
    global_step = 0

    # world reset
    if env.index == 0:
        env.reset_world()

    memory_position = 0
    update = 0

    # replay_memory     
    replay_memory = ReplayMemory(REPLAY_SIZE, SEED)

    for id in range(MAX_EPISODES):
        
        # reset
        env.reset_pose()
        
        terminal = False
        ep_reward = 0
        step = 1

        # generate goal
        env.generate_goal_point()
        
        # get_state
        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]

        # episode 1
        while not terminal and not rospy.is_shutdown():
                        
            state_list = comm.gather(state, root=0)

            ## get_action
            #-------------------------------------------------------------------------
            # generate actions at rank==0
            a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)

            # execute actions
            #-------------------------------------------------------------------------            
            real_action = comm.scatter(scaled_action, root=0)
            
            ## run action
            #-------------------------------------------------------------------------            
            env.control_vel(real_action)

            rospy.sleep(0.001)

            ## get reward
            #-------------------------------------------------------------------------
            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1

            #-------------------------------------------------------------------------

            # get next state
            #-------------------------------------------------------------------------
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            #-------------------------------------------------------------------------

            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)
            state_next_list = comm.gather(state_next, root=0)


            ## training
            #-------------------------------------------------------------------------
            if env.index == 0:

                # add data in replay_memory
                #-------------------------------------------------------------------------
                replay_memory.push(state[0], state[1],state[2],a, logprob, r_list, state_next[0], state_next[1], state_next[2], terminal_list)
                if len(replay_memory) > BATCH_SIZE:
            
                    ## update
                    #-------------------------------------------------------------------------
                    update = sac_update_stage(policy=policy, optimizer=optimizer, critic=critic, critic_opt=critic_opt, critic_target=critic_target, 
                                            batch_size=BATCH_SIZE, memory=replay_memory, epoch=EPOCH, replay_size=REPLAY_SIZE,
                                            tau=TAU, alpha=ALPHA, gamma=GAMMA, updates=update, update_interval=UPDATE_INTERVAL,
                                            num_step=BATCH_SIZE, num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []
                    global_update += 1
                    update = update

            step += 1
            state = state_next

        ## save policy
        #--------------------------------------------------------------------------------------------------------------
        if env.index == 0:
            if global_update != 0 and global_update % 1000 == 0:
                torch.save(policy.state_dict(), policy_path + '/policy_{}'.format(global_update))
                torch.save(critic.state_dict(), policy_path + '/critic_{}'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)


if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

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
        policy_path = 'policy_0809'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        critic = QNetwork(frames=LASER_HIST, action_space=2)
        critic.cuda()

        critic_opt = Adam(critic.parameters(), lr=LEARNING_RATE)
        critic_target = QNetwork(frames=LASER_HIST, action_space=2)
        critic_target.cuda()

        hard_update(critic_target, critic)

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage1_2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, critic=critic, critic_opt=critic_opt, critic_target=critic_target, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass
