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

from model.net import Actor,Critic, CNNPolicy, GaussianExploration
from td3_stage_world import StageWorld
from model.td3 import td3_update_stage
from model.td3 import generate_action, select_action

from model.replay_memory import ReplayMemory


MAX_EPISODES = 30000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 3072
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 1
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5

REPLAY_SIZE = 100000
SEED = 123456

exploration_noise = 0.1 
TAU = 0.005                 # target policy update parameter
policy_noise = 0.2          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter

MAX_ACTION = 1.0

def run(comm, env, policy, policy_path, action_bound, optimizer):

    actor, actor_target, critic_1, critic_1_target, critic_2, critic_2_target = policy
    actor_opt, critic_1_opt, critic_2_opt = optimizer

    noise = GaussianExploration(action_bound)

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

        terminal = False
        ep_reward = 0
        step = 1
        
        # generate_goal
        env.generate_goal_point()
        
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
            mean, action = select_action(env=env, state_list=state_list,
                                                         actor=actor, action_bound=action_bound)

            # exploration
            a = noise.get_action(mean, step)
            
            '''
            a = a + np.random.normal(0, exploration_noise, size=(1,2)) #action size check
            a = a.clip(action_bound[0], action_bound[1])
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
    

          
            #-------------------------------------------------------------------------
            ## save memory (replay_memory)
            if env.index == 0:
                replay_memory.push(state[0], state[1],state[2], a, r_list, state_next[0], state_next[1], state_next[2], terminal_list)
                
            step += 1
            state = state_next

        ## training 
        #------------------------------------------------------------------------------ 
        if env.index == 0:
            policy_list = [actor, actor_target, critic_1, critic_1_target, critic_2, critic_2_target]
            optimizer_list = [actor_opt, critic_1_opt, critic_2_opt]

            if len(replay_memory) > BATCH_SIZE:
                # update policy
                td3_update_stage(policy=policy_list, optimizer=optimizer_list, batch_size=BATCH_SIZE, memory=replay_memory, epoch = step, 
                                        replay_size=REPLAY_SIZE, gamma=GAMMA, num_step=BATCH_SIZE, num_env=NUM_ENV, frames=LASER_HIST, 
                                        obs_size=OBS_SIZE, act_size=ACT_SIZE, tau=TAU, policy_noise=policy_noise, noise_clip=noise_clip, policy_delay=policy_delay)
                global_update += 1

        # save policy
        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(actor.state_dict(), policy_path + '/actor_{}'.format(global_update))
                torch.save(critic_1.state_dict(), policy_path + '/critic_1_{}'.format(global_update))
                torch.save(critic_2.state_dict(), policy_path + '/critic_2_{}'.format(global_update))

                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)


if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname + 'td3'):
        os.makedirs('./log/' + hostname + 'td3')
    output_file = './log/' + hostname + 'td3' + '/output.log'
    cal_file = './log/' + hostname + 'td3' + '/cal.log'

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
        policy_path = 'policy_0819'
        
        #actor
        actor = Actor(frames=LASER_HIST, action_space=2, max_action = MAX_ACTION)
        actor.cuda()
        
        actor_opt = Adam(actor.parameters(), lr=LEARNING_RATE)
        
        actor_target = Actor(frames=LASER_HIST, action_space=2, max_action = MAX_ACTION)
        actor_target.cuda()

        actor_target.load_state_dict(actor.state_dict())


        #critic1
        critic_1 = Critic(frames=LASER_HIST, action_space=2)
        critic_1.cuda()
        
        critic_1_opt = Adam(critic_1.parameters(), lr=LEARNING_RATE)
        
        critic_1_target = Critic(frames=LASER_HIST, action_space=2)
        critic_1_target.cuda()

        critic_1_target.load_state_dict(critic_1.state_dict())
        

        #critic2
        critic_2 = Critic(frames=LASER_HIST, action_space=2)
        critic_2.cuda()
        
        critic_2_opt = Adam(critic_2.parameters(), lr=LEARNING_RATE)
        
        critic_2_target = Critic(frames=LASER_HIST, action_space=2)
        critic_2_target.cuda()

        critic_2_target.load_state_dict(critic_2.state_dict())


        mse = nn.MSELoss()

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
            logger.info('######Critic_1 Loading Model########')
            logger.info('####################################')
            state_dict = torch.load(file)
            critic_1.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('######Critic_1 Start Training########')
            logger.info('#####################################')

        file = policy_path + '/stage1_2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('######Critic_2 Loading Model########')
            logger.info('####################################')
            state_dict = torch.load(file)
            critic_2.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('######Critic_2 Training##############')
            logger.info('#####################################')


        policy_list = [actor, actor_target, critic_1, critic_1_target, critic_2, critic_2_target]
        optimizer_list = [actor_opt, critic_1_opt, critic_2_opt]


    else:
        actor = None
        actor_target = None

        critic_1 = None
        critic_1_target = None
        critic_2 = None
        critic_2_target = None

        policy_path = None
        actor_opt = None
        critic_1_opt = None
        critic_2_opt = None

        policy_list = [actor, actor_target, critic_1, critic_1_target, critic_2, critic_2_target]
        optimizer_list = [actor_opt, critic_1_opt, critic_2_opt]


    try:
        run(comm=comm, env=env, policy=policy_list, policy_path=policy_path, action_bound=action_bound, optimizer=optimizer_list)
    except KeyboardInterrupt:
        pass
