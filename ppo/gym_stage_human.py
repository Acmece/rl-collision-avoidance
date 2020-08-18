import time
import rospy
import copy
import tf
import numpy as np

from collections import deque

from geometry_msgs.msg import Twist, Pose, Point32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8


class StageWorld():
    def __init__(self, beam_num, index, num_env):
        print(index)
        self.index = index
        self.num_env = num_env
        node_name = 'Stage_human_Env_' + str(index)
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
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.
        # self.reset_pose = None

        self.init_pose = None

        self.obs_stack = None
        
        self.ctr_flag = 0
        self.ctr_pose = 0


        # for get reward and terminate
        self.stop_counter = 0

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'human_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'human_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)
        
        goal_point_topic = 'human_' + str(index) + '/pub_goal_point'
        self.pub_goal_point = rospy.Publisher(goal_point_topic, Pose, queue_size=2)


        # ---------Subscriber-----------------

        object_state_topic = 'human_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'human_'+ str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'human_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'human_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)


        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)



        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None

        while self.scan is None or self.speed is None or self.state is None or self.speed_GT is None or self.state_GT is None:
            pass
       
        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)

    def ground_truth_callback(self, GT_odometry): ##?????

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
        assert len(pose)==3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)


    def step(self):

        obs = self.get_laser_observation()
        min_obs = obs.min()
        min_index = obs.argmin()
        
        max_obs = obs.max()
          
        if min_obs > -0.35:
            self.ctr_flag = 0

        else :
            if min_index < 2:
                self.ctr_flag = 1
            else : 
                self.ctr_flag = 2
        
        state_human  = self.get_self_stateGT()
        
        is_crash = self.get_crash_state()

        random_theta = np.random.uniform(0.2, 2.6)
        random_v = np.random.uniform(0.5, 1.0)
        random_w = np.random.uniform(-0.7, 0.7)

        if is_crash:
            self.reset_pose()

        if random_w > 0.5 or random_w < -0.5:
            random_w = 0

        if self.ctr_flag == 0:

            action = [random_v,random_w]
            self.control_vel(action)
            rospy.sleep(0.001)

        elif self.ctr_flag == 1:

            action = [0.0, 0.5]
            self.control_pose([state_human[0], state_human[1], state_human[2] + random_theta])
            rospy.sleep(0.001)
        
        elif self.ctr_flag == 2:

            action = [0.0, -0.5]
            self.control_pose([state_human[0], state_human[1], state_human[2] - random_theta])
            rospy.sleep(0.001)



    def get_state(self):
        
        obs = self.get_laser_observation()
        self.obs_stack = deque([obs, obs, obs])
        goal_obs = np.asarray(self.get_local_goal())
        speed_obs = np.asarray(self.get_self_speed())
        state_obs = [self.obs_stack, goal_obs, speed_obs]

        return state_obs

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

    def generate_goal_point(self): #episode 
        [x_g, y_g] = self.generate_random_goal()
        self.goal_point = [x_g, y_g]
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)  ## pre_distance -> distance -> 1step get_reward function -> pre_distance (no matched)
        self.distance = copy.deepcopy(self.pre_distance) # zero to local goal


    def reset_pose(self):

        random_pose = self.generate_random_pose()
        rospy.sleep(0.01)
        self.control_pose(random_pose)
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        # start_time = time.time()
        while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:
            [x_robot, y_robot, theta] = self.get_self_stateGT()
            self.control_pose(random_pose)
        rospy.sleep(0.01)


    def generate_random_pose(self):

        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        dis = np.sqrt(x ** 2 + y ** 2)
        while (dis > 9) and not rospy.is_shutdown():
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            dis = np.sqrt(x ** 2 + y ** 2)
        theta = np.random.uniform(0, 2 * np.pi)
        return [x, y, theta]
