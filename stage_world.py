import time
import rospy
import copy
import tf
import random
import cv2
import numpy as np

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty

class StageWorld():
    def __init__(self, beam_num):
        rospy.init_node('StageEnv', anonymous=None)

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
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        self.object_state_sub = rospy.Subscriber('base_pose_ground_truth', Odometry, self.ground_truth_callback)
        self.laser_sub = rospy.Subscriber('base_scan', LaserScan, self.laser_scan_callback)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odometry_callback)
        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)

        # rospy.spin()
        # rospy.sleep(1)

        # # Wait until the first callback
        while self.scan is None:
            pass
        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Moving")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

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
        self.scan = np.array(scan.ranges)  # ndarray shape(660,)
        self.laser_cb_num += 1


    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 5.6  # range_max:5.6
        scan[np.isinf(scan)] = 5.6
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
        return scan_sparse / 5.6 - 0.5    # need to be figure out
        # return self.scan_param

    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

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
        x = random.uniform(-(self.map_size[0] / 2 - self.goal_size), self.map_size[0] / 2 - self.goal_size)
        y = random.uniform(-(self.map_size[1] / 2 - self.goal_size), self.map_size[1] / 2 - self.goal_size)
        self.goal_point = [x, y]
        while not self.goal_point_check() and not rospy.is_shutdown():
            x = random.uniform(-(self.map_size[0] / 2 - self.goal_size), self.map_size[0] / 2 - self.goal_size)
            y = random.uniform(-(self.map_size[1] / 2 - self.goal_size), self.map_size[1] / 2 - self.goal_size)
            self.goal_point = [x, y]
        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)

    # attribute: pre_distance, distance, goal_point

    def goal_point_check(self):
        goal_x = self.goal_point[0]
        goal_y = self.goal_point[1]
        pass_flag = True
        x_pixel = int(goal_x * self.R2P[0] + self.map_origin[0])
        y_pixel = int(goal_y * self.R2P[1] + self.map_origin[1])
        window_size = int(self.robot_size / 2 * np.amax(self.R2P))
        for x in xrange(np.amax([0, x_pixel - window_size]), np.amin([self.map_pixel[0] - 1, x_pixel + window_size])):
            for y in xrange(np.amax([0, y_pixel - window_size]),
                            np.amin([self.map_pixel[1] - 1, y_pixel + window_size])):
                if self.map[self.map_pixel[1] - y - 1, x] == 1:
                    pass_flag = False
                    break
            if not pass_flag:
                break
        if abs(goal_x) < 2. and abs(goal_y) < 2.:
            pass_flag = False
        return pass_flag


    def get_reward_and_terminate(self, t):
        terminate = False
        reset = False
        laser_scan = self.get_laser_observation()
        laser_min = np.amin(laser_scan)
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        alpha = np.arctan2(self.goal_point[1] - y, self.goal_point[0] - x) - theta

        # reward = v * np.cos(w) - 0.01
        reward = (self.pre_distance - self.distance) * np.cos(w) - 0.01
        # reward = -0.5 * 0.2
        result = 0
        if v == 0.0 and t > 10 and laser_min < 0.4 / 5.6 - 0.5:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        if self.distance < self.goal_size:
            reward = 5.
            terminate = True
            reset = True
            print 'Reach the Goal'
            result = 3
        else:
            if self.stop_counter == 2 and t <= 200:
                reward = -5.
                terminate = True
                reset = True
                print 'Crash'
                result = 2
            elif t > 200:
                terminate = True
                reset = True
                print 'Time Out'
                result = 1

        return reward, terminate, result

    def control(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)




    def render_map(self, path):
        [x, y, theta] = self.get_self_stateGT()
        self.reset_map(path)
        self.map = self.draw_point([x, y], self.robot_size, self.robot_value,
                                  self.map, self.map_pixel, self.map_origin, self.R2P)
        return self.map

    def reset_map(self, path): # path = [[0, 0], env.goal_point]
        self.map = copy.deepcopy(self.raw_map)
        goal_point = path[-1]
        self.map = self.draw_point(goal_point, self.goal_size, self.goal_value,
                                  self.map, self.map_pixel, self.map_origin, self.R2P)
        return self.map

    def draw_point(self, point, size, value, map_img, map_pixel, map_origin, R2P):
        # x range
        if not isinstance(size, np.ndarray):
            x_range = [np.amax([int((point[0] - size / 2) * R2P[0]) + map_origin[0], 0]),
                       np.amin([int((point[0] + size / 2) * R2P[0]) + map_origin[0],
                                map_pixel[0] - 1])]

            y_range = [np.amax([int((point[1] - size / 2) * R2P[1]) + map_origin[1], 0]),
                       np.amin([int((point[1] + size / 2) * R2P[1]) + map_origin[1],
                                map_pixel[1] - 1])]
        else:
            x_range = [np.amax([int((point[0] - size[0] / 2) * R2P[0]) + map_origin[0], 0]),
                       np.amin([int((point[0] + size[0] / 2) * R2P[0]) + map_origin[0],
                                map_pixel[0] - 1])]

            y_range = [np.amax([int((point[1] - size[1] / 2) * R2P[1]) + map_origin[1], 0]),
                       np.amin([int((point[1] + size[1] / 2) * R2P[1]) + map_origin[1],
                                map_pixel[1] - 1])]

        for x in xrange(x_range[0], x_range[1] + 1):
            for y in xrange(y_range[0], y_range[1] + 1):
                # if map_img[map_pixel[1] - y - 1, x] < value:
                map_img[map_pixel[1] - y - 1, x] = value
        return map_img




if __name__ == '__main__':
    env = StageWorld(40)
    # while not rospy.is_shutdown():
    #     speed = env.get_self_speedGT()
    #     state = env.get_self_stateGT()
    #     # print 'state {}, speed {}'.format(state, speed)
    #     a = env.get_laser_observation()
    #     # print 'range_min {}, range_max {}'.format(a[-2], a[-1])
    #
    #     sim_time = env.get_sim_time()
    #     print 'sim time {}'.format(sim_time)
    env.reset_world()
    a = np.asarray([0.4, np.pi/7])
    env.control(a)
    print 'out'





