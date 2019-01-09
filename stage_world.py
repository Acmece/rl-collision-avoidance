import time
import rospy
import copy
import tf
import random
import cv2
import numpy as np

from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8


class StageWorld():
    def __init__(self, beam_num, index, num_env):
        self.index = index
        self.num_env = num_env
        node_name = 'StageEnv_' + str(index)
        rospy.init_node(node_name, anonymous=None)

        self.beam_mum = beam_num
        self.laser_cb_num = 0
        self.scan = None

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.3

        self.robot_value = 10.
        self.goal_value = 0.





        # used in tatget point check
        map_img = cv2.imread('./worlds/Obstacles.jpg', 0)
        ret, binary_map = cv2.threshold(map_img, 10, 1, cv2.THRESH_BINARY)
        binary_map = 1 - binary_map
        height, width = binary_map.shape
        self.map_pixel = np.array([width, height]) # pixel shape
        self.R2P = self.map_pixel / self.map_size # corresponding pixels every meter
        self.robot_size = 0.5
        self.map = binary_map.astype(np.float32) # map
        self.raw_map = copy.deepcopy(self.map)   # for the use of resetmap
        self.map_origin = self.map_pixel / 2 - 1 # pixel coordinate

        # for get reward and terminate
        self.stop_counter = 0

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)


        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)






        # rospy.spin()
        # rospy.sleep(1)

        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:
            pass


        # while self.scan is None:
        #     pass


        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)


    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1


    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 6.0 - 0.5


    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        rospy.sleep(0.5)


    def generate_goal_point(self):
        # radians = 2 * np.pi / self.num_env * self.index + np.pi
        # x = 12 * np.cos(radians)
        # y = 12 * np.sin(radians)
        # self.goal_point = [x, y]
        # self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        # self.distance = copy.deepcopy(self.pre_distance)

        [x, y] = self.generate_random_goal()
        self.goal_point = [x, y]
        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)


    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        laser_min = np.amin(laser_scan)
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        result = 0

        # if v < 0.05 and t > 10 and laser_min < 0.4 / 6.0 - 0.5:
        #     self.stop_counter += 1
        # else:
        #     self.stop_counter = 0
        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            print 'Reach the Goal'
            result = 3
            reward_g = 15

        if is_crash == 1:
            terminate = True
            print 'Env {} Crashed'.format(self.index)
            result = 2
            reward_c = -15.

        if np.abs(w) > np.pi / 0.7:
            print 'Env {} execute too large angular speed'.format(self.index)
            reward_w = -0.1 * np.abs(w)

        if t > 399:
            terminate = True
            print 'Env {} Time out'.format(self.index)
            result = 1

        reward = reward_g + reward_c + reward_w

        # if t > 400:
        #     terminate = True
        #     print 'Env {} Time out'.format(self.index)
        #     result = 1
        # elif is_crash == 1:
        #     terminate = True
        #     print 'Env {} Crashed'.format(self.index)
        #     result = 2
        #     reward = -15.
        # elif self.distance < self.goal_size:
        #     terminate = True
        #     print 'Reach the Goal'
        #     result = 3
        #     reward = 15.
        # else:
        #     pass

        if terminate == True:
            # radians = 2 * np.pi / self.num_env * self.index
            # x = 12 * np.cos(radians)
            # y = 12 * np.sin(radians)
            pose = self.generate_random_pose()
            self.control_pose(pose)
            print 'reset pose'

        return reward, terminate, result


    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)


    def control_pose(self, pose):
        pose_cmd = Pose()
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0
        pose_cmd.orientation.x = 0
        pose_cmd.orientation.y = 0
        pose_cmd.orientation.z = 0
        pose_cmd.orientation.w = 1
        self.cmd_pose.publish(pose_cmd)

    def generate_random_pose(self):
        x = np.random.uniform(-13, 13)
        y = np.random.uniform(-13, 13)
        dis = np.sqrt(x**2 + y**2)
        while dis > 14 and not rospy.is_shutdown():
            x = np.random.uniform(-13, 13)
            y = np.random.uniform(-13, 13)
            dis = np.sqrt(x ** 2 + y ** 2)

        pose = [x, y]
        return pose

    def generate_random_goal(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(-13, 13)
        y = np.random.uniform(-13, 13)
        dis_origin = np.sqrt(x ** 2 + y ** 2)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_origin > 13 or dis_goal > 20 or dis_goal < 10) and not rospy.is_shutdown():
            x = np.random.uniform(-13, 13)
            y = np.random.uniform(-13, 13)
            dis_origin = np.sqrt(x ** 2 + y ** 2)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)

        return [x, y]







if __name__ == '__main__':
    # env_list = []
    # for i in range(12):
    #     env = StageWorld(40, index=i)
    #     env_list.append(i)

    # while not rospy.is_shutdown():
    #     speed = env.get_self_speedGT()
    #     state = env.get_self_stateGT()
    #     # print 'state {}, speed {}'.format(state, speed)
    #     a = env.get_laser_observation()
    #     # print 'range_min {}, range_max {}'.format(a[-2], a[-1])
    #
    #     sim_time = env.get_sim_time()
    #     print 'sim time {}'.format(sim_time)


    # env.reset_world()
    # a = np.asarray([0.4, np.pi/7])
    # for _ in range(1000):
    #     for env in env_list:
    #         env.control(a)
    #         # rospy.sleep(0.1)
    # print 'out'



    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    action = None

    env = StageWorld(512, index=rank)
    # if rank == 0:
    #     # env.reset_world()
    #     a = np.repeat(np.asarray([0.4, np.pi / 7])[np.newaxis], 12, axis=0)
    # else:
    #     a = None
    #
    # env.generate_goal_point()
    # print '{} and {}'.format(rank, env.goal_point)

    a = np.repeat(np.asarray([0.4, np.pi / 7])[np.newaxis], 12, axis=0)

    for i in range(400):

        stateGT = env.get_self_stateGT()

        send_data = stateGT
        recv_data = comm.gather(send_data, root=0)
        if rank ==0:
            print recv_data
            print len(recv_data)
            print len(recv_data)
            if i == 100:
                pose = np.asarray([6,6])
                env.control_pose(pose)
            if i == 200:
                pose = np.asarray([0,0])
                env.control_pose(pose)
            if i == 300:
                pose = np.asarray([-6, -6])
                env.control_pose(pose)


        action = comm.scatter(a, root=0)
        env.control_vel(action)
        rospy.sleep(0.001)






