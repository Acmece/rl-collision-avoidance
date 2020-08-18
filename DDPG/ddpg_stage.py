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

from model.net import Actor,Critic
from ddpg_stage_world import StageWorld
from model.ddpg import ddpg_update_stage
from model.ddpg import generate_action

from model.replay_memory import ReplayMemory
from model.utils import hard_update


MAX_EPISODES = 120000
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
CRITIC_LEARNING_RATE = 1e-3
ACTOR_LEARNING_RATE = 1e-4

REPLAY_SIZE = 100000
SEED = 123456

exploration_noise = 0.1 
noise_clip = 0.5

TAU = 0.001

MAX_ACTION = 1.0

def run(comm, env, policy, policy_path, action_bound, optimizer):

    actor, actor_target, critic, critic_target = policy
    actor_opt, critic_opt = optimizer

    # rate = rospy.Rate(5)
    buff = []
    global_update = 0
    global_step = 0

    replay_memory = ReplayMemory(REPLAY_SIZE, SEED)

    #world reset
    if env.index == 0:
        env.reset_world()


    for id in range(MAX_EPISODES):
        
        #reset
        env.reset_pose()

        env.generate_goal_point()
        terminal = False
        ep_reward = 0
        step = 1

        # get_state
        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]

        
        while not terminal and not rospy.is_shutdown():
        
            state_list = comm.gather(state, root=0)

            ## get_action
            #-------------------------------------------------------------------------
            # generate actions at rank==0
            mean, a = generate_action(env=env, state_list=state_list,
                                                         actor=actor, action_bound=action_bound)
            
            '''
            noise = np.random.normal(0, exploration_noise, size=(1,2)) #action size check
            a = mean + noise
            

            a = action.clip(action_bound[0], action_bound[1])
            '''


            # execute actions
            real_action = comm.scatter(a, root=0)
            #-------------------------------------------------------------------------            
            
            ### step ############################################################
            ## run action
            env.control_vel(real_action)
            #-------------------------------------------------------------------------

            # rate.sleep()
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


            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)
            #-------------------------------------------------------------------------

            ########################################################################

            state_next_list = comm.gather(state_next, root=0)
    
            ## training
            #-------------------------------------------------------------------------
            if env.index == 0:
                replay_memory.push(state[0], state[1], state[2], a, r_list, state_next[0], state_next[1], state_next[2], terminal_list)              
                policy_list = [actor, actor_target, critic, critic_target]
                optimizer_list = [actor_opt, critic_opt]

                if len(replay_memory) > BATCH_SIZE:
                    ddpg_update_stage(policy=policy_list, optimizer=optimizer_list, batch_size=BATCH_SIZE, memory=replay_memory, epoch = EPOCH, 
                                            replay_size=REPLAY_SIZE, gamma=GAMMA, num_step=BATCH_SIZE, num_env=NUM_ENV, frames=LASER_HIST, 
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE, tau=TAU)
                    global_update += 1

            step += 1
            state = state_next

    
        if env.index == 0:
            if global_update != 0 and global_update % 3000 == 0:
                torch.save(actor.state_dict(), policy_path + '/actor_{}'.format(global_update))
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
    if not os.path.exists('./log/' + hostname + 'ddpg'):
        os.makedirs('./log/' + hostname + 'ddpg')
    output_file = './log/' + hostname + 'ddpg' + '/output.log'
    cal_file = './log/' + hostname + 'ddpg' + '/cal.log'

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
    action_bound = [[0, -1], [1, 1]] 
    
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy_0812_reward_g'
        
        #actor
        actor = Actor(frames=LASER_HIST, action_space=2, max_action = MAX_ACTION)
        actor.cuda()
        
        actor_opt = Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
        
        actor_target = Actor(frames=LASER_HIST, action_space=2, max_action = MAX_ACTION)
        actor_target.cuda()

        hard_update(actor_target, actor)

        #critic
        critic = Critic(frames=LASER_HIST, action_space=2)
        critic.cuda()
        
        critic_opt = Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)
        
        critic_target = Critic(frames=LASER_HIST, action_space=2)
        critic_target.cuda()

        hard_update(critic_target, critic)


        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        
        file = policy_path + '/stage1_2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('#######Actor Loading Model##########')
            logger.info('####################################')
            state_dict = torch.load(file)
            actor.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('#######Actor Start Training##########')
            logger.info('#####################################')
        
        file = policy_path + '/stage1_2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('######Critic Loading Model########')
            logger.info('####################################')
            state_dict = torch.load(file)
            critic.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('######Critic Start Training########')
            logger.info('#####################################')

        policy_list = [actor, actor_target, critic, critic_target]
        optimizer_list = [actor_opt, critic_opt]


    else:
        actor = None
        actor_target = None

        critic= None
        critic_target = None


        policy_path = None
        actor_opt = None
        critic_opt = None


        policy_list = [actor, actor_target, critic, critic_target]
        optimizer_list = [actor_opt, critic_opt]


    try:
        run(comm=comm, env=env, policy=policy_list, policy_path=policy_path, action_bound=action_bound, optimizer=optimizer_list)
    except KeyboardInterrupt:
        pass
