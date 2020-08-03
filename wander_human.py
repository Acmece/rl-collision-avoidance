import os
import logging
import sys
import socket
import numpy as np
import rospy

from mpi4py import MPI

from collections import deque

from gym_stage_human import StageWorld

MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 6
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5

def run(env):
        
    while True:
    
        env.step()
        
        rospy.sleep(0.001)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    print("go")

    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    
    print("ENV")
 
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:

        print('####################################')
        print('############wander Start########$###')
        print('####################################')
        
    try:
        run(env=env)
    except KeyboardInterrupt:
        pass
