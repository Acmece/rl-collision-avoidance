#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2015, Open Source Robotics Foundation, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Brian Gerkey

import sys
import threading
import time
import unittest

from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Odometry
import rospy
import rostest
import tf.transformations

class TestStageRos(unittest.TestCase):

    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)
        rospy.init_node('pose_tester', anonymous=True)

    def _base_pose_ground_truth_sub(self, msg):
        self.base_pose_ground_truth = msg

    def _odom_sub(self, msg):
        self.odom = msg

    def setUp(self):
        self.odom = None
        self.base_pose_ground_truth = None
        self.done = False
        self.odom_sub = rospy.Subscriber('odom', Odometry, self._odom_sub)
        self.base_pose_ground_truth_sub = rospy.Subscriber(
            'base_pose_ground_truth', Odometry, self._base_pose_ground_truth_sub)
        # Make sure we get base_pose_ground_truth
        while self.base_pose_ground_truth is None:
          time.sleep(0.1)
        # Make sure we get odom and the robot is stopped (not still moving
        # from the previous test). We can count on stage to return true zeros.
        while (self.odom is None or
               self.odom.twist.twist.linear.x != 0.0 or
               self.odom.twist.twist.linear.y != 0.0 or
               self.odom.twist.twist.linear.z != 0.0 or
               self.odom.twist.twist.angular.x != 0.0 or
               self.odom.twist.twist.angular.y != 0.0 or
               self.odom.twist.twist.angular.z != 0.0):
          time.sleep(0.1)

    def _pub_thread(self, pub, msg):
        while not self.done:
            pub.publish(msg)
            time.sleep(0.05)

    # Test that, if we command the robot to drive forward for a while, that it does
    # so.
    def test_cmdvel_x(self):
        odom0 = self.odom
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        twist = Twist()
        twist.linear.x = 1.0
        # Make a thread to repeatedly publish (to overcome Stage's watchdog)
        t = threading.Thread(target=self._pub_thread, args=[pub, twist])
        t.start()
        time.sleep(3.0)
        odom1 = self.odom
        self.done = True
        t.join()
        # Now we expect the robot's odometric pose to differ in X but nothing
        # else
        self.assertGreater(odom1.header.stamp, odom0.header.stamp)
        self.assertNotAlmostEqual(odom1.pose.pose.position.x, odom0.pose.pose.position.x)
        self.assertAlmostEqual(odom1.pose.pose.position.y, odom0.pose.pose.position.y)
        self.assertAlmostEqual(odom1.pose.pose.position.z, odom0.pose.pose.position.z)
        self.assertAlmostEqual(odom1.pose.pose.orientation.x, odom0.pose.pose.orientation.x)
        self.assertAlmostEqual(odom1.pose.pose.orientation.y, odom0.pose.pose.orientation.y)
        self.assertAlmostEqual(odom1.pose.pose.orientation.z, odom0.pose.pose.orientation.z)
        self.assertAlmostEqual(odom1.pose.pose.orientation.w, odom0.pose.pose.orientation.w)

    # Test that, if we command the robot to turn in place for a while, that it does
    # so.
    def test_cmdvel_yaw(self):
        odom0 = self.odom
        pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        twist = Twist()
        twist.angular.z = 0.25
        # Make a thread to repeatedly publish (to overcome Stage's watchdog)
        t = threading.Thread(target=self._pub_thread, args=[pub, twist])
        t.start()
        time.sleep(3.0)
        odom1 = self.odom
        self.done = True
        t.join()
        # Now we expect the robot's odometric pose to differ in yaw (which will
        # appear in the quaternion elements z and w) and not elsewhere
        self.assertGreater(odom1.header.stamp, odom0.header.stamp)
        self.assertAlmostEqual(odom1.pose.pose.position.x, odom0.pose.pose.position.x)
        self.assertAlmostEqual(odom1.pose.pose.position.y, odom0.pose.pose.position.y)
        self.assertAlmostEqual(odom1.pose.pose.position.z, odom0.pose.pose.position.z)
        self.assertAlmostEqual(odom1.pose.pose.orientation.x, odom0.pose.pose.orientation.x)
        self.assertAlmostEqual(odom1.pose.pose.orientation.y, odom0.pose.pose.orientation.y)
        self.assertNotAlmostEqual(odom1.pose.pose.orientation.z, odom0.pose.pose.orientation.z)
        self.assertNotAlmostEqual(odom1.pose.pose.orientation.w, odom0.pose.pose.orientation.w)

    # Test that, if we command the robot to jump to a pose, it does so.
    def test_pose(self):
        pub = rospy.Publisher('cmd_pose', Pose, queue_size=1)
        while pub.get_num_connections() == 0:
            time.sleep(0.1)
        pose = Pose()
        pose.position.x = 42.0
        pose.position.y = -42.0
        pose.position.z = 142.0
        roll = 0.2
        pitch = -0.3
        yaw = 0.9
        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        pub.publish(pose)
        time.sleep(3.0)
        # Now we expect the robot's ground truth pose to be what we told, except
        # for z, roll, and pitch, which should all be zero (Stage is 2-D, after all).
        bpgt = self.base_pose_ground_truth
        self.assertAlmostEqual(bpgt.pose.pose.position.x, pose.position.x)
        self.assertAlmostEqual(bpgt.pose.pose.position.y, pose.position.y)
        self.assertEqual(bpgt.pose.pose.position.z, 0.0)
        q = [bpgt.pose.pose.orientation.x,
             bpgt.pose.pose.orientation.y,
             bpgt.pose.pose.orientation.z,
             bpgt.pose.pose.orientation.w]
        e = tf.transformations.euler_from_quaternion(q)
        self.assertEqual(e[0], 0.0)
        self.assertEqual(e[1], 0.0)
        self.assertAlmostEqual(e[2], yaw)

    # Test that, if we command the robot to jump to a pose (with a header), it does so.
    def test_pose_stamped(self):
        pub = rospy.Publisher('cmd_pose_stamped', PoseStamped, queue_size=1)
        while pub.get_num_connections() == 0:
            time.sleep(0.1)
        ps = PoseStamped()
        ps.header.frame_id = 'ignored_value'
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = -42.0
        ps.pose.position.y = 42.0
        ps.pose.position.z = -142.0
        roll = -0.2
        pitch = 0.3
        yaw = -0.9
        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        ps.pose.orientation.x = q[0]
        ps.pose.orientation.y = q[1]
        ps.pose.orientation.z = q[2]
        ps.pose.orientation.w = q[3]
        pub.publish(ps)
        time.sleep(3.0)
        # Now we expect the robot's ground truth pose to be what we told, except
        # for z, roll, and pitch, which should all be zero (Stage is 2-D, after all).
        bpgt = self.base_pose_ground_truth
        self.assertAlmostEqual(bpgt.pose.pose.position.x, ps.pose.position.x)
        self.assertAlmostEqual(bpgt.pose.pose.position.y, ps.pose.position.y)
        self.assertEqual(bpgt.pose.pose.position.z, 0.0)
        q = [bpgt.pose.pose.orientation.x,
             bpgt.pose.pose.orientation.y,
             bpgt.pose.pose.orientation.z,
             bpgt.pose.pose.orientation.w]
        e = tf.transformations.euler_from_quaternion(q)
        self.assertEqual(e[0], 0.0)
        self.assertEqual(e[1], 0.0)
        self.assertAlmostEqual(e[2], yaw)

NAME = 'stage_ros'
if __name__ == '__main__':
    rostest.unitrun('stage_ros', NAME, TestStageRos, sys.argv)
